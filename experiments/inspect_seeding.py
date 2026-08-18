"""
Diagnostic: find every constructor signature + every hardcoded PRNGKey/seed
in the ccka package, so we can patch seeding precisely instead of guessing.

Run this INSIDE your project's environment (where `ccka` is importable):
    python inspect_ccka_seeding.py
"""
import inspect

print("="*70)
print("1) Constructor signatures")
print("="*70)
from ccka.aligner.kta import fullKTA, centroidBasedKTA, quackKTA, randomKTA, greedyKTA
for cls in [fullKTA, randomKTA, greedyKTA, quackKTA, centroidBasedKTA]:
    print(f"\n{cls.__name__}.__init__{inspect.signature(cls.__init__)}")

print()
print("="*70)
print("2) KernelModel / circuit weight-init signature")
print("="*70)
from ccka.models.kernel import KernelModel
from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
print(f"quackEmbeddingCircuit.__init__{inspect.signature(quackEmbeddingCircuit.__init__)}")
try:
    print(f"quackEmbeddingCircuit.init_weights{inspect.signature(quackEmbeddingCircuit.init_weights)}")
except Exception as e:
    print(f"  (couldn't inspect init_weights: {e})")

print()
print("="*70)
print("3) grep for hardcoded PRNGKey / np.random.seed inside ccka source")
print("="*70)
import ccka, os, subprocess
pkg_dir = os.path.dirname(ccka.__file__)
print(f"Package dir: {pkg_dir}\n")
try:
    out = subprocess.run(
        ["grep", "-rn", "-E", "PRNGKey|np.random.seed|default_rng|random.seed",
         pkg_dir],
        capture_output=True, text=True,
    )
    print(out.stdout or "(no matches found)")
except Exception as e:
    print(f"grep failed: {e}")