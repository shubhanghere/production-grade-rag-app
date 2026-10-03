"""
Create or update the LangSmith dataset used by the eval harness.

Run as:   python -m scripts.build_dataset
Requires: LANGSMITH_API_KEY in the environment (or .env).

Idempotent:
  - Creates the dataset `wms-cloud-26b-golden` if it does not exist.
  - Upserts one example per GoldenExample, keyed by its stable `id`, so
    re-running updates questions/answers/metadata in place rather than
    creating duplicates.
"""

from dotenv import load_dotenv
from langsmith import Client

from eval.golden_set import GOLDEN_SET, RELEASE

# Scripts must load .env for the LangSmith key — the app server does this in
# main.py; without it the client falls back to the (keyless) shell env → 401.
load_dotenv(override=True)

DATASET_NAME = "wms-cloud-26b-golden"
DATASET_DESCRIPTION = (
    f"Golden Q&A set for the WMS Cloud {RELEASE} documentation RAG app. "
    "Balanced across definitional / configuration / api_reference query types."
)


def _get_or_create_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        return client.read_dataset(dataset_name=DATASET_NAME)
    return client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )


def build() -> None:
    client = Client()
    dataset = _get_or_create_dataset(client)

    # Index existing examples by our stable `id` (stored in metadata) so we can
    # decide create-vs-update per example.
    existing = {
        ex.metadata.get("example_id"): ex
        for ex in client.list_examples(dataset_id=dataset.id)
        if ex.metadata and ex.metadata.get("example_id")
    }

    created, updated = 0, 0
    for ex in GOLDEN_SET:
        inputs = {"question": ex.question}
        outputs = {"reference_answer": ex.reference_answer}
        metadata = {
            "example_id": ex.id,
            "query_type": ex.query_type,
            "release": ex.release,
            "gold_section_id": ex.gold_section_id,  # None until ingestion fills it in
        }

        if ex.id in existing:
            client.update_example(
                example_id=existing[ex.id].id,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            updated += 1
        else:
            client.create_example(
                dataset_id=dataset.id,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            created += 1

    print(
        f"Dataset '{DATASET_NAME}': {created} created, {updated} updated "
        f"({len(GOLDEN_SET)} total)."
    )


if __name__ == "__main__":
    build()
