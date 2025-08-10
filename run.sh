source .venv/bin/activate

python3 main.py --backend pennylane --config configs/extended/checkerboard.yaml
python3 main.py --backend pennylane --config configs/extended/checkerboard_quack.yaml

python3 main.py --backend pennylane --config configs/extended/corners.yaml
python3 main.py --backend pennylane --config configs/extended/corners_quack.yaml

python3 main.py --backend pennylane --config configs/extended/double_cake.yaml
python3 main.py --backend pennylane --config configs/extended/double_cake_quack.yaml

python3 main.py --backend pennylane --config configs/extended/moons.yaml
python3 main.py --backend pennylane --config configs/extended/moons_quack.yaml

python3 main.py --backend pennylane --config configs/extended/donuts.yaml
python3 main.py --backend pennylane --config configs/extended/donuts_quack.yaml

echo "Thank You !"