import numpy as np

class ArtifactProsodyScorer:
    def __init__(self, config=None):
        self.config = config
        
        # Heuristic Defaults (if not in config)
        self.norm_flatness = 0.5
        self.norm_centroid = 2000.0
        self.norm_f0_std = 50.0
        
    def normalize_artifact(self, raw_scores):
        # raw: {flatness, centroid_std}
        # higher flatness -> synthetic? (or silence)
        # higher centroid var -> natural? 
        # This logic is tricky without data analysis.
        # User prompt: "artifact_score... derived from signal artifacts"
        # Let's assume High Score = Suspicious/Synthetic.
        
        # Synthetic often has unnatural flatness or weird high-freq artifacts.
        s1 = raw_scores['flatness'] # 0..1 already roughly
        s2 = raw_scores['centroid_std'] / self.norm_centroid # Normalizing
        
        # Composite score
        # Example: High flatness = bad. Low centroid var = bad?
        
        score = (s1 + (1.0 - min(s2, 1.0))) / 2.0
        return float(score)

    def normalize_prosody(self, raw_scores):
        # raw: {f0_std, jitter_proxy}
        # High stability => likely bonafide => score should be 'HIGH STABILITY' ?
        # Prompt: "prosody_stability... high stability => likely bonafide"
        # So Score 1.0 = Stable (Real), Score 0.0 = Unstable (Spoof/Replay)
        
        # Spoof/VC often has monotone or jittery pitch.
        
        f0_std = raw_scores['f0_std']
        
        # If std is normal (human range ~ 20-100Hz depending on intonation), score is high
        # If std is near 0 (robotic monotone), score is low.
        
        normalized_std = min(f0_std / self.norm_f0_std, 1.0)
        
        return float(normalized_std)
