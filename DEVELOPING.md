
python .\src\expression_recognition\models\face_mesh_adapter.py preview --draw --cam 2

in src:
python -m expression_recognition.models.smile_preview --cam 0 --draw

export PYTHONPATH="$PWD/src"

Running the Annotations Script:
python src/expression_recognition/data/annotations.py \
  --classes neutral,smile,frown \
  --session my_session \
  --landmarks \
  --auto-pack \
  --T 12 \
  --stride 6 \
  --val-split 0.2 \
  --draw

