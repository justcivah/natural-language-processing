import pandas as pd
from openai import OpenAI
import time
from pydantic import BaseModel


class TranslationResult(BaseModel):
    translations: list[str]


# return a sublist of size batch_size
def batch_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# get single batch trnaslations
def translate_batch(client, batch, max_retries=5, retry_delay=2):
    prompt = (
        "Translate the following English sentences into Italian. "
        "Return the translations in the same order as a JSON object with an array named translations."
        "Return only valid JSON, with no extra text.\n\n"
        + "\n".join(batch)
    )

    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model="gpt-5-mini",
                input=prompt,

                # force the model to return a structured object
                text_format=TranslationResult
            )

            result = response.output_parsed
            print(f"Translated {len(result.translations)} sentences")

            if isinstance(result.translations, list) and len(result.translations) == len(batch):
                return result.translations

        except Exception as e:
            print(f"Exception on attempt {attempt+1}: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return [""] * len(batch)


def translate_dataset(file_path, batch_size):
    df = pd.read_csv(file_path)

    if "italian" not in df.columns:
        df["italian"] = ""

    sentences = df["english"].astype(str).tolist()
    client = OpenAI()

    # start from last untranslated row
    italian = df["italian"]
    is_nonempty = italian.notna() & italian.str.strip().ne("")
    nonempty_idx = df.index[is_nonempty]

    if len(nonempty_idx) > 0:
        start_idx = nonempty_idx[-1] + 1
    else:
        start_idx = 0

    for batch_start in range(start_idx, len(sentences), batch_size):
        batch_end = min(batch_start + batch_size, len(sentences))
        batch = sentences[batch_start:batch_end]

        print(f"Processing {file_path}: rows {batch_start}-{batch_end - 1}")

        translations = translate_batch(client, batch)
        df.loc[batch_start:batch_end - 1, "italian"] = translations

        # save progress after every batch
        df.to_csv(file_path, index=False)


batch_size = 20

translate_dataset("datasets/dataset_a/test.csv", batch_size)
translate_dataset("datasets/dataset_a/valid.csv", batch_size)
translate_dataset("datasets/dataset_a/train.csv", batch_size)

translate_dataset("datasets/dataset_b/test.csv", batch_size)
translate_dataset("datasets/dataset_b/valid.csv", batch_size)
translate_dataset("datasets/dataset_b/train.csv", batch_size)