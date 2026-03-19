import os
import re
import time
import random
import pandas as pd
import polars as pl
from math import ceil
from typing import Optional
from vllm import LLM, SamplingParams
from collections import Counter
import typer
import kaggle_evaluation.aimo_2_inference_server

app = typer.Typer()


@app.command()
def main(
    input_csv: str = typer.Option(..., "--input_csv", help="Input CSV file path"),
    output_csv: str = typer.Option(..., "--output_csv", help="Output CSV file path"),
    logdir: Optional[str] = typer.Option(None, "--logdir", help="Unused, MavicBF does not provide any logs"),
    model_path: str = typer.Option(..., "--model_path", help="Path to the model"),
    tensor_parallel_size: int = typer.Option(4, "--tensor_parallel_size", help="Tensor parallel size"),
    max_num_seqs: int = typer.Option(90 * 2**3, "--max_num_seqs", help="Maximum number of sequences"),
    max_model_len: int = typer.Option(4096 * 4, "--max_model_len", help="Maximum model length"),
    trust_remote_code: bool = typer.Option(True, "--trust_remote_code/--no-trust_remote_code", help="Trust remote code"),
    gpu_memory_utilization: float = typer.Option(0.95, "--gpu_memory_utilization", help="GPU memory utilization"),
    seed: int = typer.Option(1488, "--seed", help="Random seed"),
    temperature: float = typer.Option(1.0, "--temperature", help="Sampling temperature"),
    skip_special_tokens: bool = typer.Option(True, "--skip_special_tokens/--no-skip_special_tokens", help="Skip special tokens in output"),
    max_tokens: int = typer.Option(4096, "--max_tokens", help="Maximum tokens to generate"),
    stop: Optional[str] = typer.Option("</think>", "--stop", help="Stop sequence (comma-separated if multiple)"),
):
    # This also had no CLI support at all, so add it here
    print(f"Arguments: input_csv={input_csv}, output_csv={output_csv}, logdir={logdir}, model_path={model_path}, tensor_parallel_size={tensor_parallel_size}")

    MAX_NUM_SEQS = max_num_seqs
    MAX_MODEL_LEN = max_model_len
    llm_model_pth = model_path
    # start_time = time.time()
    # cutoff_time = start_time + float('inf')
    # cutoff_time = time.time() + (4 * 60 + 57) * 60
    global_start_time = time.time()

    # Parse stop sequences
    stop_sequences = [s.strip() for s in stop.split(",")] if stop else []

    llm = LLM(
        llm_model_pth,
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        # enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()


    def extract_boxed_text(text):
        pattern = r"oxed{(.*?)}"
        matches = re.findall(pattern, text)
        return matches[-1] if matches else ""


    def select_answer(answers):
        counter = Counter()
        for a in answers:
            try:
                if int(a) == float(a):
                    counter[int(a)] += 1 + random.random() / 1e3
            except:
                pass
        if counter:
            return sorted([(v, k) for k, v in counter.items()], reverse=True)[0][1] % 1000
        return 3


    def create_starter_messages(question, index):
        options = []
        for _ in range(2):
            options.append(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
                    },
                    {
                        "role": "user",
                        "content": question
                        + " Return final answer within \\boxed{}, after taking modulo 1000.",
                    },
                ]
            )
        for _ in range(2):
            options.append(
                [
                    {
                        "role": "system",
                        "content": "You are the most powerful math expert. Please solve the problems with deep reasoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000.",
                    },
                    {"role": "user", "content": question},
                ]
            )
        options.append(
            [
                {
                    "role": "system",
                    "content": "You are a helpful and harmless math assistant. You should think step-by-step and you are good at reverse thinking to recheck your answer and fix all possible mistakes. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}.",
                },
                {"role": "user", "content": question},
            ]
        )
        return options[index % len(options)]


    def predict_for_question(question: str) -> int:
        question_start_time = time.time()

        selected_questions_only = True
        print(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
        print(question)

        num_of_iters = 4
        num_of_seqs = 90

        messages = [
            create_starter_messages(question, index) for index in range(num_of_seqs)
        ]
        list_of_texts = [
            tokenizer.apply_chat_template(
                conversation=message, tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]
        old = list_of_texts
        new = list_of_texts
        res = []

        for k in range(num_of_iters):
            sampling_params = SamplingParams(
                temperature=temperature,  # randomness of the sampling
                # min_p=0.05,
                # top_p=0.9,
                skip_special_tokens=skip_special_tokens,  # Whether to skip special tokens in the output
                max_tokens=max_tokens,
                stop=stop_sequences,
            )
            outputs = llm.generate(
                old,
                sampling_params=sampling_params,
            )
            new = []

            new_old = []
            for i in range(len(old)):
                if extract_boxed_text(outputs[i].outputs[0].text):
                    res.append(old[i] + outputs[i].outputs[0].text)
                else:
                    new_old.append(old[i] + outputs[i].outputs[0].text)

            # if num_of_iters == 3 and k == 1 and len(new_old) > 130:
            #     curr_num = min(20, len(new_old))

            #     new_old = random.sample(new_old, curr_num)
            print(list(extract_boxed_text(x) for x in res))
            for i in range(len(new_old)):
                for _ in range(2):
                    new.append(new_old[i])
            old = new

            if len(res) >= 50:
                numbers = []
                for r in res:
                    try:
                        num = int(extract_boxed_text(r))
                        numbers.append(num)
                    except ValueError:
                        pass

                if len(numbers) >= 50:
                    counter = Counter(numbers)
                    total = len(numbers)

                    for num, cnt in counter.items():
                        if cnt >= ceil(total * 0.8):
                            question_end_time = time.time()
                            print(num)
                            print(
                                "TIME FOR QUESTION: ",
                                question_end_time - question_start_time,
                            )
                            print("PRE RESULT!!!")
                            return num

        answer = select_answer(list(extract_boxed_text(x) for x in res))
        print(answer)
        question_end_time = time.time()
        print("TIME FOR QUESTION: ", question_end_time - question_start_time)
        return answer


    def predict(id_: pl.DataFrame, question: pl.DataFrame):
        _id = id_.item(0)
        q = question.item(0)
        ans = predict_for_question(q)
        print(f"{_id}: {ans}")
        return pl.DataFrame({"id": _id, "answer": ans})


    # pd.read_csv('./reference.csv').to_csv('reference.csv', index=False)
    df = pd.read_csv(input_csv)

    # Drop 'answer' column if it exists
    if "answer" in df.columns:
        df = df.drop("answer", axis=1)

    df.to_csv("reference.csv", index=False)

    inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(
        predict
    )
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(("reference.csv",))

    # Kaggle output server writes to ./submission.parquet -> Extract and write to CSV specified by output_csv
    pl.read_parquet("submission.parquet").write_csv(output_csv)


if __name__ == "__main__":
    app()