from src.ansatze import SomeAnsatz, SimpleAnsatz0, SimpleAnsatz1, HardcodedTwirledSimpleAnsatz0
from src.embeddings import RXEmbedding, RXEmbeddingWEnt
from src.losses import binary_cross_entropy_loss

circuit_dict = {'RXEmbedding': RXEmbedding,
                'RXEmbeddingWEnt': RXEmbeddingWEnt,
                'SomeAnsatz': SomeAnsatz,
                'SimpleAnsatz0': SimpleAnsatz0,
                'SimpleAnsatz1': SimpleAnsatz1,
                'HardcodedTwirledSimpleAnsatz0': HardcodedTwirledSimpleAnsatz0}
loss_dict = {"bce_loss": binary_cross_entropy_loss}
