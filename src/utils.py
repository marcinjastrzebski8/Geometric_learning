from src.ansatze import SomeAnsatz, SomeAnsatzTwirled
from src.embeddings import rx_embedding, rx_w_ent_embedding
from src.losses import binary_cross_entropy_loss

circuit_dict = {'rx_embedding': rx_embedding,
                'rx_w_ent_embedding': rx_w_ent_embedding,
                'SomeAnsatz': SomeAnsatz,
                'SomeAnsatzTwirled': SomeAnsatzTwirled}
loss_dict = {"bce_loss": binary_cross_entropy_loss}
