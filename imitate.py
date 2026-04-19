
from pathlib import Path
from BC import BehaviorCloningAgent

BATCH_SIZE = 16
EPOCHS = 5
N_GAMES = 2

for game in range(1, N_GAMES + 1):
    print(f"\n=== Imitating game {game}/{N_GAMES} ===")
    dataset_path = Path(f"datasets/winning_{game}.pt")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}, skipping...")
        continue

    agent = BehaviorCloningAgent(
        dataset_path=dataset_path,
        checkpoint_path="checkpoints/bc_agent.pt",
    )
    agent.learn(batch_size=BATCH_SIZE, epochs=EPOCHS)