import jax
import jax.numpy as jnp
from flax import nnx
from model import Transformer

N_LAYERS = 8
D_MODEL = 128
VOCAB_SIZE = 13
SEQ_LEN = 16 # ABC+DEF=GHI
FFN_DIM = 512
N_HEADS = 4
HEAD_DIM = 32


def generate_data(num_samples, key):
    key_a, key_b = jax.random.split(key, 2)
    a = jax.random.randint(key_a, (num_samples,), minval=0, maxval=1000)
    b = jax.random.randint(key_b, (num_samples,), minval=0, maxval=1000)

    c = a + b

    PLUS = 10
    EQUALS = 11
    SPACE = 12

    def get_digits(num_array):
        hundreds = num_array//100
        tens = (num_array//10) % 10
        ones = num_array % 10
        return jnp.stack([hundreds, tens, ones], axis = 1)
        
    a_digits = get_digits(a)
    b_digits = get_digits(b)

    thousands = c // 1000
    hundreds = (c // 100) % 10
    tens = (c // 10) % 10
    ones = c % 10

    c = jnp.stack([thousands, hundreds, tens, ones], axis = 1)

    plus = jnp.full((num_samples,), PLUS, dtype = jnp.int32)
    equals = jnp.full((num_samples,), EQUALS, dtype = jnp.int32)

    pad = jnp.full((num_samples,5), SPACE, dtype = jnp.int32)
    
    pass

def main():
    key = jax.random.PRNGKey(18)
    rngs = nnx.Rngs(params = key)
    model = Transformer(
        num_layers=N_LAYERS,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        ffn_dim=FFN_DIM,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        rngs=rngs
    )
    


if __name__ == "__main__":
    main()
