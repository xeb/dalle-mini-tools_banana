import os
import base64
import subprocess
from dalle_mini_tools.generate import Generator

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    # device = 0 if torch.cuda.is_available() else -1
    model = Generator()

def postprocessing(script_path:str, run_name:str):
    cmds = [f"{script_path}", run_name]
    wd = os.path.join(os.getcwd(), "dalle-mini-tools")
    print(f"Using {wd=}")
    p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=wd)
    out, err = p.communicate()
    if p.returncode != 0:
        print(f"Exception\n{err=}\n\n{out=}")
    print(f"Postprocess complete for {run_name=}")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:   
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = model.generate(prompt=prompt)
    print(f"Return {result=}")

    print(os.getcwd())

    postprocessing("./postprocess.sh", result)

    result_obj = {
        "runname": result,
    }

    final = os.path.join(result, "final.png")
    if os.path.exists(final):
        with open(final, "rb") as f:
            result_obj["final_b64"] = base64.b64encode(f.read())

    # Return the results as a dictionary
    return result_obj
