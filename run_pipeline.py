import subprocess, sys

STEPS = [
    'src/renderer/schematic_renderer.py',
    'src/renderer/realistic_renderer.py',
    'src/evaluation/ground_truth_generator.py',
    'src/evaluation/question_generator.py',
    'src/evaluation/model_evaluator.py',
]

for step in STEPS:
    print(f'\n>>> Running {step} ...')
    res = subprocess.run([sys.executable, step], env={**__import__('os').environ, 'PYTHONPATH': '.'})
    if res.returncode != 0:
        sys.exit(res.returncode)
print('\n=== RailSwitch End-to-End Pipeline Complete! ===')
