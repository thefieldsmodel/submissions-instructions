# Singularity Container Demo

## A Standard for Using Singularity Containers

These guidelines outline the criteria your container has to adhere to so we can easily evaluate it. In order to facilitate reproducibility and to ensure that you can support you with a minimal amount of manual interventions on our side, we will make use of **Singularity** containers (which are similar to Docker, but is built on namespaces and thus avoids root exposure; it is standard for high-performance scientific computing) and a strict rules on how these need to be submitted to us. There are different platforms for Singularity containers, we use [SingularityCE](https://sylabs.io/singularity/).

Please note that we cannot give you direct access to our cluster; rather you submit the container (as well as some metadata that we outlined below, that helps us assess your code to ensure that no malicious code is executed on the cluster) and we run it on your behalf and return run results (if you simply want to evalute a very big model) or train results (if you want to train a model) to you. Since we may have to deal with *many* containers, so **we will not have the time for an extensive back and forth with you to get your container to work**, which makes it necessary to strictly adhere to the rules outlined below in terms of what the container needs to contain, and how it is to be access.

We mandate one set of rules to _run_ whatever is in the container and another set of standards to _train_ whatever is in the container. These different sets of rules and associate container files we will refer to as "_run_ mode" and "_train_ mode" 

You can use "run mode", for example, to test (a particularly big model, or set of models) on a benchmark. You can use "train mode" to for all kinds of training runs (supervised learning, reinforcement learning, etc). Note that if you want to get checkpoints to make manual decisions (e.g., to test by hand whether the model performs well), you need to submit the first container for training -> we return the result to you, that would be you checkpoint -> you submit another container for training -> we return again the result to you, etc. Checkpointing models thus counts, from our perspective, as submitting distinct containers, where each must satify all the rules outlined here; we wouldn't have the capacity to maintain an ongoing dialogue with you to allow "introspection" on a single container run.

Please make sure you try the container in both "_run_ mode" and "_train_ mode" before you send it to use (either using a commercial vendor that has Singularity support, which is the container we will use---more on that below---or on your own system). 

In order to separate the code from the data, you have to provide, additionally to the container (in either "run mode" or "train mode") a [presigned AWS S3 URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html) URL `and `upload.py` scripts. We will use the URL and download everything that is there (most commonly weights, and potentially datasets), and then push the trained artefacts or results back to you. The next section outines how to transfer files to and from us.


## 1. Storage Management

We will solely us S3 buckets to manage file transfer. 

#### Download
Your S3 bucket presigned URL must contains the artefacts as described below in the "run mode" or "train mode" sections

We will download all artifacts required to run your submission from the URL you provide us with. The artefacts are different whether you want run a (arge) model or train a model -- see below. 

#### Upload
You need to provide us with an `upload.py` script. Its  CLI interface is defined as follows:
* `s3_url`: The presigned URL to your AWS S3 bucket for either upload or download.
* `source_dir`: Path to the source (weights in most cases) that are to be uploaded to S3.

## 2. Running the Pipeline ("run mode")

#### Run mode main items

You will provide four items when executing the container in _run mode_:
1. A **`README.md` file** that documents at a high level how your pipeline works at inference, when we run the container, and which parameters may affect the quality of the solution.
2. A **`input.csv` file** with columns `id;problem` (answer optional!) whose rows are the problems, questions, or any other text you want to run through you model.
3. A **Singularity image** that we will execute on your behalf in _run_ mode.
4. The exact Singularity **build file** you used to create the Singularity image. This will allow us to see which dependencies, versions, drivers etc. you have used to get a clear picture of the solution and implications when running it outside of the container. Below we will detail how the build file must look like.

The build file and the final image file must both be named `<pipelinename_run_YYYYMMDD>` file, where "YYYYMMDD" is the submission day. Thus, a set of example files that make up a full "run mode" submission could look like the following, if your pipeline consists of majority voting:
* `README.md`
* `majority-voting_run_20261703.sif`
* `majority-voting_run_20261703.def`
* `input.csv`
* all model weights required to run the container


#### The build file: main structure

The build file needed to build the Singularity image needs to contain the following:
* All dependencies pre-installed, as well as any system dependencies, CUDA versions, etc.
* All environment variables are set up as needed
* A single Python script that is the entry point of the image (excluding CLI wrappers like Typer), with all arguments being passed to it.

See `majority-voting_run_20261703.def` for a short example of a Singularity build file that fits these requirements by pulling the relevant information (the "layers" of the Docker container) from a Docker container with pre-built PyTorch and CUDA versions, copies all source code from a `src` directory into the Singularity container, sets various environment variables, creates various folders and installs all dependencies from the `requirements.txt`, and then exposes `/app/run.py` as the entrypoint, and `run.py` receives all CLI arguments. 

#### The build file: the `run.py` file

We will run `/app/run.py` and we expect 3 **required** CLI arguments:

* `--model_path`: Local path to the downloaded base model weights.
* `--input_csv` to specify a local path on the host (the cluster) to a CSV with columns `id;problem` (the column `answer` is optional!)
* `--output_csv` to specify a local path on the host (the cluster) respective to the working directory of the container.
* `--logdir` to specify a logging path respective to the working directory of the container.
	
**All** other parameters that serve to customize your model(s) must be CLI options **with default values** that have **clear names per parameter**.

Your `run.py`file **must** allow for configuration of at least the following parameters as CLI options:
 * `num_gpus`: Number of GPUS to use
 * `num_ctx`: Context length of the model
 * `max_new_tokens`: Maximum new tokens per model completion.
 * `temperature`: Temperature of the sampling process.
 * `top_p`: Top-p for restricting decoding to tokens with probability `> p`.
 * `top_k`: Top-k for restricting decoding to the `k` most tokens with the highest probability.

The `run.py` script must implement a function `run` that takes all of these parameters, propagates them to the LLMs that come up in your pipeline and runs the complete pipeline. In case your pipeline consists of multiple LLMs (e.g. if you implemented a "[council of LLMs](https://github.com/karpathy/llm-council)"), each parameter needs to take in a list of values that are propagated to each LLM your pipeline uses.

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
* We strictly require the output to be written to the path specified by `output_csv` with at least the columns `id;prediction`. 
* The pipeline should not fail if the path does not yet exist (i.e., call `Path(...).mkdir(parentes=True, exist_ok=True)`)
* Additional columns are permitted and even encouraged if they give information about the prediction (if it solved the problem when an `answer` key is additionally given, token count, etc.)


##  3. Training a Model ("train mode")

While the previous "run mode" outlined how one could run an entire **inference pipeline** that is inside a container, here we describe how one could setup a pipeline to **train a model**.

#### Train mode main items

You will provide four items when executing the container in _train mode_:
1. A `README.md`file that documents at a high level how your training pipeline works
2. A dataset (e.g. in parquet files).
3. A built Singularity image that we will execute on your behalf in _train_ mode.
4. The exact Singularity build file you used to create the Singularity image. This will allow us to see which dependencies, versions, drivers etc. you have used to get a clear picture of the solution and implications when running it outside of the container. We will detail below how the build file must look like.

The build file and the final image file must both be named `<pipelinename_train_YYYYMMDD>` file, where "YYYYMMDD" is the submission day. Thus, a set of example files that make up a full _train mode_ submission would be:
* `README.md`
* `sft-lora_train_20261703.sif`
* `sft-lora_train_20261703.def`
* all model weights required to run the container
* the files making up the dataset you wish to train on, as well as any files you may want to use to auto-benchmark as you train.

#### The build file: main structure

The build file needed to build the Singularity image needs to contain the following:
* All dependencies pre-installed, as well as any system dependencies, CUDA versions, etc.
* All environment variables are set up as needed
* A single Python script that is the entry point of the image (excluding CLI wrappers like Typer), with all arguments being passed to it.

See `sft-lora_train_20261703.def` for a short example of a Singularity build file that fits these requirements by pulling the relevant information (the "layers" of the Docker container) from a Docker container with pre-built PyTorch and CUDA versions, copies all source code from a `src` directory into the Singularity container, sets various environment variables, creates various folders and installs all dependencies from the `requirements.txt`, and then exposes `/app/train.py` as the entrypoint, and `train.py` receives all CLI arguments. 

#### The build file: the `train.py` file

We will run `/app/train.py` and we expect 3 **required** CLI arguments:

* `--model_path`: Local path to the downloaded base model weights. 
* `--output_path`: Local path where the trained weights should be written to.
* `--dataset_path`: Local path to the downloaded dataset used for the training process.

**All** other parameters that serve to customize your model and training pipeline must be CLI options **with default values** that have **clear names per parameter**.

Below are a few parameters that are commonly used to configure how a model acts or samples its responses (see also above in _run mode_):
* `num_ctx`: Context length of the model
* `max_new_tokens`: Maximum new tokens per model prompt.
* `temperature`: Temperature of the sampling process
* `top_p`:  Top-p for restricting decoding to tokens with probability `> p`.
* `top_k`: Top-k for restricting decoding to the `k` most tokens with the highest probability.
* ...

Below are a few suggested training parameters, that may aid in quicker deployment of parameters sweeps or avoid
you having to re-build the container for minor changes.

* `num_gpus`: Number of GPUS to use, although this highly depends on the training requirements. Multi-Node training will need more sophisticated configuration options.
* `learning_rate`: Base Learning rate for training.
* `num_train_epochs`: Epochs to train for.
* `per_device_batch_size`: Batch-Size per GPU, relevant for DDP training.
* `gradient_accumulation_steps`: Defines over how many steps the gradients should be accumulated.
* ...


#### Training on a cluster
For training on clusters, we will possibly require more configuration options to enable multi-node and multi-GPU training. However, most of these could be set via environment variables inside of the submitted image directly instead of using CLI parameters, as they are static and highly dependent on the designated workflow. 


## Code Format

This section is relevant for all code that is written and submitted by you, not only the main script. Maintaining this will both ensure your code is readable and consistent across submissions, should they be open-sourced.

If you use multiple modules to structure your code, be aware that all of them should fit the coding style.

* Place **all** imports at the top, except if you really need imports in certain places when using multiprocessing (this can happen for torch, for example)
* Refrain from using relative imports (i.e., don't use `.some_packge` etc.), IF you need to use modular imports.
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
* Use doc strings to describe what each functions does and what parameters/returns are expected.
