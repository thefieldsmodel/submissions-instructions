# Singularity Container Guidelines 

## Overview of the Submission to the Fields Model Initiative

These guidelines outline how the Fields Model can run and/or train models for you. In order to facilitate 1) reproducibility and 2) to ensure that we can support you with a minimal amount of manual intervention on our side, we will make use of **Singularity** containers, strict rules and standardized interfaces on how these need to be submitted to us. The Fields Model is a joint effort by Benchmarks+Baselines and compute partners, with the LLMC group at the NII as main compute partner.

Please note that the Fields Model cannot give you direct access to the compute partner's cluster. Rather, you make a submission (which contains a Singularity container as well as some other files we describe below) through a Google Form, administered by Benchmarks+Baselines, and the LLMC runs this on your behalf and returns run results (if you want to evaluate a very big model) or train results (if you want to train a model) back to you. 

Due to a possibly *large number* of containers, we at the Fields Model **will not have the time for an extensive back and forth with you to get your container to work**, which makes it necessary to strictly adhere to the rules and standardized interfaces outlined below in terms of what the container needs to contain, and how it is to be accessed. 

The submission that you make via the Google form contains a presigned AWS S3 URL (see below) that points us to:
* a container package that contains everything we need to run your model (a Singularity container, a README file, weights, etc.; more information below).
* a Python upload script so we can transfer results back to you.

Our workflow consists of three actions:
1. Download the container package + upload script at the URL you provided
2. Run a single Python file in your container (our rules specify exactly the interface of that Python file)
3. Run the Python upload script in order to return whatever the container generated back to you.

We use Singularity containers because that is the preferred format of our compute partner. These are similar to Docker, but are built on namespaces and thus avoid root exposure; it is standard for high-performance scientific computing. There are different platforms for Singularity containers, we use [SingularityCE](https://sylabs.io/singularity/).

There are two types of Singularity containers you can submit: Containers designed to run a pipeline of LLMs contained inside it, or containers designed to train models inside it. We use the word "pipeline" since you may not be confined to a single LLM, but rather use multiple ones in conjunction (e.g. if you implemented a "[council of LLMs](https://github.com/karpathy/llm-council)").

We mandate one set of rules to _run_ whatever is in the container and another set of rules to _train_ whatever is in the container. These different sets of rules and associated container files we will refer to as "_run_ variant" and "_train_ variant".

You can use the "run variant", for example, to test (a particularly big model, or set of models) on a benchmark. You can use the "train variant" for all kinds of training runs (supervised learning, reinforcement learning, etc). Note that if you want to get checkpoints to make manual decisions (e.g., to test by hand whether the model performs well), you need to make a first submission with a container in "train variant" -> we return the result to you, which would be your checkpoint -> you make another submission of a container in "train variant" -> we return again the result to you, which would be your second checkpoint, and so on. The checkpointing models thus counts, from our perspective, as making distinct submissions, where each must satisfy all the rules outlined here; we wouldn't have the capacity to maintain an ongoing dialogue with you to allow "introspection" on a single container run.

Before you make a submission, please make sure you test your container before you send it to us (either using a commercial vendor that has Singularity support). 

The next sections outline the precise rules and standardized interface for the URL (section 1), the container package (section 2a for containers in the "run variant" and section 2b for containers in the "train variant") and the upload script to transfer files to and from us (section 3). We will rely on Python scripts with prescribed CLI interface, containing positional as well as named arguments, as these make it easy for us to simply pass the right arguments to your script without having to spend time touching the code. I.e., we would like to simply run `python yourscript.py arg1 arg2`, where we prescribe the arguments you expose.

If you cannot follow these in detail, we will not be able to proceed to run or train your model.

## 1. The URL

We will solely use [S3 buckets](https://aws.amazon.com/s3/) to manage file transfer. This is well-maintained and comparatively cheap way to host very large files (e.g., hundreds of GB of model weights). It is your responsibility to maintain this and to cover any associated costs -- which should be minimal compared to costs you would incur when training models yourself.

You should use a [presigned AWS S3 URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html). This is a secure, time-limited link that you can share with us, that will ensure that your submission artefacts cannot be discovered online.

We will download all artefacts required to run your submission from the URL you provide us with. 

## 2a. Container package in "run variant"

Below we describe the exact contents of the container package. These are only valid for the container in the run variant. For a container in train variant (see section 2b) the contents are slightly different.

Note that containers in both variants are not stateful, i.e., files written inside the container will be gone once it is "down" again after running. Thus, the container should write its final artefacts that are to be uploaded to the host -- and the paths should be reflected in `upload.py`. The container will be able to access the file system of the host, and our interface is designed so that we can pass the appropriate locations to the container.

#### Run variant main items

Your container package will consist of five items when executing the container in _run variant_:
1. A **`README.md` file** that documents at a high level how your pipeline works at inference, when we run the container, and which parameters may affect the quality of the solution.
2. A **`input.csv` file** with columns `id,problem` (answer optional!) whose rows are the problems, questions, or any other text you want to run through your model.
3. A **Singularity image** that we will execute on your behalf in _run_ variant.
4. The exact Singularity **build file** you used to create the Singularity image. This will allow us to see which dependencies, versions, drivers etc. you have used to get a clear picture of the solution and implications when running it outside of the container. Below we will detail how the build file must look like.
5. All weights (including necessary metadata, the tokenizer config, chat templates, etc.) that are required by your pipeline and other data of all models that make up your pipeline.

Everything except these five items should be placed inside the container! This will ensure that the container will not get unduly large, by containing weights, etc.
We mandate to use the `input.csv` file externally, as that helps, in case the model is released to provide an easy interface for third-parties to change the input.csv` to evaluate the model pipeline -- as opposed to a more embedded approach where the problems and questions are contained inside the container. 

The container build file and the resulting container image file must both be named `<pipelinename>_run_<YYYYMMDD>`, where "YYYYMMDD" is the submission day. Thus, a set of example files that make up a full "run variant" submission could look like the following, if your pipeline consists of majority voting the outputs of a single LLM, for example [gpt-oss-120b model](https://huggingface.co/openai/gpt-oss-120b):
* `/README.md`
* `/input.csv`
* `/majority-voting_run_20260317.sif`
* `/majority-voting_run_20260317.def`
* `/model/files-and-weights-that-make-gpt-oss-120b`


#### The build file: main structure

The build file needed to build the Singularity image needs to contain the following:
* All dependencies pre-installed, as well as any system dependencies, CUDA versions, etc.
* All environment variables are set up as needed
* A single Python script that is the entry point of the image, with all arguments being passed to it.

See `majority-voting_run_20260317.def` for a short example of a Singularity build file that fits these requirements by pulling the relevant information (the "layers" of the Docker container) from a Docker container with pre-built PyTorch and CUDA versions, copies all source code from a `src` directory into the Singularity container, sets various environment variables, creates various folders and installs all dependencies from the `requirements.txt`, and then exposes `/app/run.py` as the entrypoint, and `run.py` receives all CLI arguments. 

#### The build file: the `run.py` file

Our sole action is to run `/app/run.py` and we expect 4 **required** CLI arguments:

* `--model_path`: a local path on the host (the cluster) to the downloaded base model weights.
* `--input_csv`:  a local path on the host (the cluster) to a CSV with columns `id,problem` (the column `answer` is optional!)
* `--output_csv`: a local path on the host (the cluster) respective to the working directory of the container.
* `--logdir`: a local path on the host (the cluster) to the working directory of the container, where logs are placed.

Your `run.py` file **must** additionally allow for configuration of at least the following parameters as CLI options in order to control inference:
 * `num_ctx`: Context length of the model
 * `max_new_tokens`: Maximum new tokens per model completion.
 * `temperature`: Temperature of the sampling process.
 * `top_p`: Top-p for restricting decoding to tokens with probability `> p`.
 * `top_k`: Top-k for restricting decoding to the `k` tokens with the highest probability.

 Your `run.py` file *may* also implement the parameters below -- i.e., if you choose to make them accessible via this script, you need to follow the naming below.  These runtime parameters may further aid the evaluation of your model by a third party by having more control over inference:
  * `num_gpus`: Number of GPUs to use for a model.
  * min_new_tokens: Prevents the model from stopping too early
  * `min_p`: Drops very low-probability tokens relative to the top token.

Of course you are free to add even further parameters here, using your preferred naming scheme (as long as the don't conflict with the above). **All** these other parameters that serve to customize your model(s), and how it runs, must be CLI options **with default values** that have **clear names per parameter**.

The `run.py` script must implement a function `run` that takes all of these parameters, propagates them to the LLMs that come up in your pipeline and runs the complete pipeline. In case your pipeline consists of multiple LLMs (e.g. if you implemented a "[council of LLMs](https://github.com/karpathy/llm-council)"), each parameter above should accept a list of values, one for each model. You should document inside the code which parameter corresponds to which model.

If you referenced a `requirements.txt` (or any other package management) inside the build file, also provide us with those files as well!

Below is a very short example of how your code should be structured sequentially to make a nicely written `run.py` file that satisfies these requirements:
```python
# ALL IMPORTS
import os
from pathlib import Path

import polars as pl
import typer
from vllm import SamplingParams, LLM

# CONSTANTS
N_GPUS = 8
PROMPT_TEMPLATE = """....{}...."""


# ALL FUNCTIONS REQUIRED TO RUN
def _apply_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text)


def predict(problem_id: int, problem: str) -> tuple[str, dict]:
    prompt = _apply_prompt(problem)

    # placeholder
    answer = "dummy_answer"
    metadata = {"id": problem_id}

    return answer, metadata


def run(
	model_path: Path = typer.Argument(..., help="Weights for the model to run inference on"),
    input_csv: Path = typer.Argument(..., help="CSV input with columns id,problem"),
    output_csv: Path = typer.Argument(..., help="Directory where output CSV will be written"),
    logdir: Path = typer.Argument(..., help="Directory for logs"),
    temperature: float = typer.Option(0.95, "--temperature", "-t", help="Sampling temperature"),
	... # Any other arguments
):
	... # Your implementation that writes to `output_csv`
	for problem_id, problem in input_df:
		answer_i, meta_i = predict(problem_id, problem)
	...

if __name__ == "__main__":
    typer.run(run)
```

#### Output Format
* We strictly require the output to be written to the path specified by `output_csv` with at least the columns `id,prediction`. 
* The pipeline should not fail if the path does not yet exist (i.e., call `Path(...).mkdir(parents=True, exist_ok=True)`)
* Additional columns are permitted and even encouraged if they give information about the prediction (if it solved the problem when an `answer` key is additionally given, token count, etc.)


## 2b. Container package in "train variant"

While the previous "run variant" outlined how to submit a container package that inside the container runs an entire **inference pipeline**, here we describe how one could  **train a model** inside the container (or multiple models, acting together to make up the pipeline that is described in the run variant, in Section 2a).

#### Train variant main items

You will provide five items when executing the container in _train variant:
1. A **`README.md`** file that documents at a high level how your training pipeline works
2. A dataset (e.g. in parquet files).
3. A **Singularity image** that we will execute on your behalf in _train_ variant.
4. The exact Singularity **build file** you used to create the Singularity image. This will allow us to see which dependencies, versions, drivers etc. you have used to get a clear picture of the solution and implications when running it outside of the container. Below we will detail how the build file must look like.
5. All weights (including necessary metadata, the tokenizer config, chat templates, etc.) that are required by your pipeline and other data you want to train on.

Everything except these five items should be placed inside the container! This will ensure that the container will not get unduly large, by containing weights, etc. 
Note that if you use internal benchmarks, e.g.  to assess for how long you train, these will now not need to conform anymore to the container specification in the run variant, as these will not be intended to be used by third parties. unlike in the run variant of the container.

The container build file and the resulting container image file must both be named `<pipelinename>_train_<YYYYMMDD>` file, where "YYYYMMDD" is the submission day. Thus, a set of example files that make up a full "train variant" submission, where you fine-tune a single LLM on your dataset, could look like the following:
* `/README.md`
* `/dataset/`
* `/sft-lora_train_20260317.sif`
* `/sft-lora_train_20260317.def`
* `/model/files-and-weights-that-make-up-the-model-you-want-to-fine-tune`

#### The build file: main structure

The build file needed to build the Singularity image needs to contain the following:
* All dependencies pre-installed, as well as any system dependencies, CUDA versions, etc.
* All environment variables are set up as needed
* A single Python script that is the entry point of the image (excluding CLI wrappers like Typer), with all arguments being passed to it.

See `sft-lora_train_20260317.def` for a short example of a Singularity build file that fits these requirements by pulling the relevant information (the "layers" of the Docker container) from a Docker container with pre-built PyTorch and CUDA versions, copies all source code from a `src` directory into the Singularity container, sets various environment variables, creates various folders and installs all dependencies from the `requirements.txt`, and then exposes `/app/train.py` as the entrypoint, and `train.py` receives all CLI arguments. 

#### The build file: the `train.py` file

Our sole action is to run `/app/train.py` and we expect 3 **required** CLI arguments:

* `--model_path`: a local path on the host (the cluster) to the downloaded base model weights. 
* `--dataset_path`: a local path on the host (the cluster) to the downloaded dataset used for the training process.
* `--output_path` :a local path on the host (the cluster) where the trained weights should be written to.
* `--logdir`: a local path on the host (the cluster) to the working directory of the container, where logs are placed.

 Your `train.py` file *may* also implement the parameters below -- i.e., if you choose to make them accessible via this script, you need to follow the naming below. These training parameters may aid in quicker deployment of parameters sweeps or avoid you having to re-build the container for minor changes:
 * `num_gpus`: Number of GPUS to use, although this highly depends on the training requirements. Multi-Node training will need more sophisticated configuration options.
* `learning_rate`: Base Learning rate for training.
* `num_train_epochs`: Epochs to train for.
* `per_device_batch_size`: Batch-Size per GPU, relevant for DDP training.
* `gradient_accumulation_steps`: Defines over how many steps the gradients should be accumulated.

Of course you are free to add even further parameters here, using your preferred naming scheme (as long as the don't conflict with the above). **All** these other parameters that serve to customize your model, and how you train it, must be CLI options **with default values** that have **clear names per parameter**. 

The `train.py` script must implement a function `train` that takes all of these parameters, and uses them to train a model (or models) inside your container. In case you train multiple models at the same time (e.g. training a teacher and a student concurrently, see https://arxiv.org/abs/2505.15034), each parameter above should accept a list of values, one for each model. You should document inside the code which parameter corresponds to which model.

If you referenced a `requirements.txt` (or any other package management) inside the build file, also provide us with those files as well!

For training on clusters, potentially more configuration options to enable multi-node and multi-GPU training will be required. However, most of these could be set via environment variables inside of the submitted image directly instead of using CLI parameters, as they are static and highly dependent on the designated workflow. 


## 3. The Upload Script

You need to provide us with an `upload.py` script located under the URL. Its CLI interface is defined as follows:
* `--s3_url`: A presigned URL to your AWS S3 bucket to which we should upload the result. Please use as default value the correct URL to which your script will upload, so you do not need to communicate the upload URL separately to us, but can embed it in the script. This can be the same URL as the download URL (be careful about overwriting files in this case) or it can be a different URL. The typical case should be that we should not have to set a custom URL by hand, and having this parameter is only for emergencies, where your provided URL does not work.
* `--source_dir`: Path to the source (weights in most cases) on the local host that are to be uploaded to the S3 bucket.



## Code Format

This section is relevant for all code that is written and submitted by you, not only the main script. Maintaining this will both ensure your code is readable and consistent across submissions, should they be open-sourced.

If you use multiple modules to structure your code, be aware that all of them should fit the coding style.

* Ensure that you catch all relevant errors and handle them gracefully; we won't have the time to debug.
* Ensure that no input from us is required when running your code (in particular your upload scripts); all relevant values (such as upload URLs) should be saved in the code.
* Place **all** imports at the top, except if you really need imports in certain places when using multiprocessing (this can happen for torch, for example).
* Refrain from using relative imports (i.e., don't use `.some_package` etc.), IF you need to use modular imports.
* Input **ONLY exactly the code required to run** your submission, nothing else (no legacy code, versions, unused functions, ...)
* Place all **arguments** below the imports (we suggest using argparse or typer for simplicity)
* Place all **CONSTANTS** directly below the imports and comment what they control.
* Use type annotations.
* Define **ALL** functions before calling any code.
* If you pull out code from your Kaggle notebook: Your code must **run completely detached from the Kaggle Core** (or any other Kaggle dependency) package when submitted to us:
* All log outputs must be saved to the `logdir` specified as CLI argument.
* **Document all failure modes** that could occur (timeouts, ...) and handle them gracefully.


#### Nice-to-haves
* To allow for easier debugging from our side, **log as much information as possible**. We propose you use `loguru` or some other easy-to-set-up logging library for convenience.
* Implement good error handling + logging in all parts of your pipeline.
* Add as many configurable arguments as possible that sensibly configure your pipeline (for example, context windows, reasoning steps, ...) that we can use to configure your solution.
* Use docstrings to describe what each functions does and what parameters/returns are expected.
