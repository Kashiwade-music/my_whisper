# my_whisper

## how to use uv
### Create a virtual environment:
- create virtual env
  - `uv venv`
- activate
  - `source .venv/bin/activate`
- install package
  - `uv pip install scipy` 
- lock env
  - `uv pip freeze | uv pip compile - -o requirements.txt`

## env set up
- create virtual env
  - `uv venv`
- activate
  - `source .venv/bin/activate`
- install package
  - `uv pip install -r requirements.txt`