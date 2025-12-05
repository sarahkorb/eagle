from openai import OpenAI
import pandas as pd
import numpy as np
import os
from copy import deepcopy
from dotenv import load_dotenv
import pickle

load_dotenv()
class EagleRanker():

    class Embedder:
        def __init__(self, api_key: str, cache_path="embedding_cache.pkl"):
            self.client = OpenAI(api_key=api_key)
            self.model_name = "text-embedding-3-large"
            self.cache_path = cache_path

            # Load cache if exists, else empty dict
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            else:
                self.cache = {}

        def embed(self, text: str): #adding caching to avoid recomputing embeddings
            if text in self.cache:
                return self.cache[text]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )

            emb = response.data[0].embedding

            self.cache[text] = emb
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)

            return emb

    def __init__(self, models: list[str], P=0.5, N=30, K=32): #from paper
        self.models = models
        self.K = K                        #
        self.P = P                         
        self.N = N                       
        self.prompts = {}                 
        self.global_elo = {model: 100.0 for model in models}
        self.local_elo = None
        self.embedder = self.Embedder(api_key=os.getenv("OPENAI_API_KEY"))


    def populate_prompts(self, df: pd.DataFrame):
  
        for idx, row in df.iterrows():
            print("on id: ", idx)
            score_big  = row["gpt-4o-2024-08-06/score"]
            score_small = row["gpt-4o-mini-2024-07-18/score"]

            if score_big > score_small:
                winner = "gpt-4o-2024-08-06"
                loser  = "gpt-4o-mini-2024-07-18"
            elif score_small > score_big:
                winner = "gpt-4o-mini-2024-07-18"
                loser  = "gpt-4o-2024-08-06"
            else:
                winner = None
                loser = None
            print("winner: ", winner) 
            text_for_embed = row["prompt"] + " " + row["question"]
            print("Now embedding text: ", text_for_embed[:20])
            embedding = self.embedder.embed(text_for_embed)

            print("now adding to prompts dict")

            self.prompts[idx] = {
                "embedding": np.array(embedding, dtype=float),
                "winner": winner,
                "loser": loser
            }

    def get_E(self, R_player, R_opponent):
        return 1 / (1 + 10 ** ((R_opponent - R_player) / 400))

    def update_elo(self, elo_dict, winner, loser, score):
        expected_score = self.get_E(elo_dict[winner], elo_dict[loser])
        elo_dict[winner] += self.K * (score - expected_score)

    def train_global_elo(self, prompt_ids): #for ALL prompts, compare models 
        for pid in prompt_ids:
            data = self.prompts[pid]
            w, l = data["winner"], data["loser"]

            if w is None:
                # tie case
                a, b = self.models #hardcoded for now to be 2
                self.update_elo(self.global_elo, a, b, 0.5)
                self.update_elo(self.global_elo, b, a, 0.5)
            else:
                self.update_elo(self.global_elo, w, l, 1) #updating elo for winners and losers
                self.update_elo(self.global_elo, l, w, 0)
    
    def cosine_similarity(self, a, b): #helper
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return np.dot(a, b) / denom

    def get_nearest_neighbors(self, embedding):
        sim_scores = []
        for pid, entry in self.prompts.items():
            sim_score = self.cosine_similarity(embedding, entry["embedding"])
            sim_scores.append((pid, sim_score))

        sim_scores.sort(key=lambda x: x[1], reverse=True) #closest neighbors 
        return [pid for pid, _ in sim_scores[:self.N]] #rreturn list of neighbors 

    def get_local_elo(self, test_embedding): #update model scores using NN prompts, not all 
        self.local_elo = deepcopy(self.global_elo)
        neighbor_ids = self.get_nearest_neighbors(test_embedding)
        for pid in neighbor_ids: #for eahc neighbor 
            data = self.prompts[pid] 
            w, l = data["winner"], data["loser"] #gte who wone and lost 
            if w is None:
                a, b = self.models #update model ELO score
                self.update_elo(self.local_elo, a, b, 0.5) #update local elo (same as global but jsut for neighbors)
                self.update_elo(self.local_elo, b, a, 0.5)
            else:
                self.update_elo(self.local_elo, w, l, 1)
                self.update_elo(self.local_elo, l, w, 0)

    def get_combined_score(self, model):
        return self.P * self.global_elo[model] + (1 - self.P) * self.local_elo[model]


    def make_model_prediction(self, prompt_text):
        emb = np.array(self.embedder.embed(prompt_text), dtype=float)
        self.compute_local_elo(emb)
        scores = {m: self.get_combined_score(m) for m in self.models}
        model_list = list(self.models)
        if len(model_list) == 2:  
            m1, m2 = model_list
            if abs(scores[m1] - scores[m2]) < 1e-12:  # safe float comparison
                return ("tie", m1)
        return max(scores, key=scores.get)
    


