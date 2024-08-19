from src.ansatze import SimpleAnsatz0, SimpleAnsatz1, HardcodedTwirledSimpleAnsatz0, GeneralCascadingAnsatz
from src.embeddings import RXEmbedding, RXEmbeddingWEnt, RotEmbeddingWEnt
from src.losses import binary_cross_entropy_loss, binary_cross_entropy_loss_jit

circuit_dict = {'RXEmbedding': RXEmbedding,
                'RXEmbeddingWEnt': RXEmbeddingWEnt,
                'RotEmbeddingWEnt': RotEmbeddingWEnt,
                'SimpleAnsatz0': SimpleAnsatz0,
                'SimpleAnsatz1': SimpleAnsatz1,
                'GeneralCascadingAnsatz': GeneralCascadingAnsatz,
                'HardcodedTwirledSimpleAnsatz0': HardcodedTwirledSimpleAnsatz0}
loss_dict = {"bce_loss": binary_cross_entropy_loss,
             "bce_loss_jit": binary_cross_entropy_loss_jit}
