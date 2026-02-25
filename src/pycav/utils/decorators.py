#This file contains the decorators needed for different tasks, such as verifications of the input and outputs
from functools import wraps

def validation_probe(func):
    @wraps(func) 
    def wrapper(self, *args, **kwargs):
        # 1. On exécute l'init (ou la fonction décorée)
        result = func(self, *args, **kwargs)
        
        # 2. On effectue les vérifications après l'assignation
        if self.fs <= 0:
            raise ValueError("The sampling frequency must be defined positive.")
        
        # 3. TRÈS IMPORTANT : On retourne le résultat
        return result
        
    # 4. TRÈS IMPORTANT : On retourne l'objet fonction wrapper
    return wrapper