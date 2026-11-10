"""
Unit tests for database repository methods and schema constraints.
"""

from pathlib import Path
import pytest

from backend.app.db.connection import get_db_connection
from backend.app.db.repository import (
    complete_training_run,
    count_generations,
    count_models,
    count_samples,
    create_generation,
    create_model,
    create_sample,
    create_training_run,
    delete_generation,
    delete_model,
    delete_sample,
    get_active_training_run,
    get_generation,
    get_model,
    get_model_by_name,
    get_sample,
    get_training_run,
    init_db,
    list_generations,
    list_models,
    list_samples,
    list_training_runs,
    update_sample,
    update_sample_transcript,
    update_training_run_progress,
)


@pytest.fixture
def test_db(tmp_path: Path):
    db_file = tmp_path / "test_repo.db"
    init_db(db_path=db_file)
    return db_file


def test_samples_crud_and_polish_unicode(test_db):
    polish_text = "Cześć! Zażółć gęślą jaźń — próba mikrofonu 123."
    sample = create_sample(
        name="Sample 1",
        filename="sample1.wav",
        file_path="/data/samples/raw/sample1.wav",
        duration_seconds=5.25,
        sample_rate=44100,
        transcript=polish_text,
        transcript_status="completed",
        db_path=test_db,
    )

    assert sample["id"] is not None
    assert sample["name"] == "Sample 1"
    assert sample["transcript"] == polish_text
    assert sample["duration_seconds"] == 5.25

    # Retrieve by ID
    retrieved = get_sample(sample["id"], db_path=test_db)
    assert retrieved is not None
    assert retrieved["transcript"] == polish_text

    # Update transcript
    updated_text = "Zaktualizowany polski tekst: ąęłńóśźż."
    updated = update_sample_transcript(sample["id"], updated_text, status="manual", db_path=test_db)
    assert updated["transcript"] == updated_text
    assert updated["transcript_status"] == "manual"

    # List and Count
    samples_list = list_samples(db_path=test_db)
    assert len(samples_list) == 1
    assert count_samples(db_path=test_db) == 1

    # Delete
    deleted = delete_sample(sample["id"], db_path=test_db)
    assert deleted is True
    assert count_samples(db_path=test_db) == 0


def test_models_crud(test_db):
    model = create_model(
        name="polish_voice_f5",
        engine="f5-tts",
        checkpoint_path="/data/models/f5_tts/polish_voice_f5.pt",
        parameters={"epochs": 10, "batch_size": 2, "learning_rate": 0.0001},
        sample_count=5,
        db_path=test_db,
    )

    assert model["name"] == "polish_voice_f5"
    assert model["engine"] == "f5-tts"
    assert model["parameters"]["epochs"] == 10

    by_name = get_model_by_name("polish_voice_f5", db_path=test_db)
    assert by_name is not None
    assert by_name["id"] == model["id"]

    models_list = list_models(engine="f5-tts", db_path=test_db)
    assert len(models_list) == 1

    delete_model(model["id"], db_path=test_db)
    assert count_models(db_path=test_db) == 0


def test_training_runs_lifecycle(test_db):
    run = create_training_run(
        engine="rvc",
        model_name="singer_rvc",
        config={"epochs": 20, "batch_size": 4},
        total_epochs=20,
        total_steps=200,
        db_path=test_db,
    )

    assert run["status"] == "queued"
    assert run["total_epochs"] == 20

    active = get_active_training_run(db_path=test_db)
    assert active is not None
    assert active["id"] == run["id"]

    # Update progress
    updated = update_training_run_progress(
        run["id"],
        current_epoch=5,
        current_step=50,
        current_loss=0.452,
        progress_percent=25.0,
        status_message="Epoch 5/20 in progress",
        db_path=test_db,
    )
    assert updated["current_epoch"] == 5
    assert updated["progress_percent"] == 25.0
    assert updated["status"] == "running"

    # Complete run
    completed = complete_training_run(run["id"], status="completed", db_path=test_db)
    assert completed["status"] == "completed"
    assert completed["completed_at"] is not None

    # No more active runs
    assert get_active_training_run(db_path=test_db) is None


def test_generations_foreign_key_set_null(test_db):
    sample = create_sample(
        name="Ref Sample",
        filename="ref.wav",
        file_path="/data/samples/raw/ref.wav",
        db_path=test_db,
    )

    gen = create_generation(
        type="tts",
        model_id="mod_123",
        output_audio_path="/data/generated/tts/out.wav",
        output_filename="out.wav",
        reference_sample_id=sample["id"],
        input_text="Dzień dobry świecie",
        parameters={"speed": 1.0},
        db_path=test_db,
    )

    assert gen["reference_sample_id"] == sample["id"]
    assert gen["parameters"]["speed"] == 1.0

    # Delete sample -> reference_sample_id in generation should become NULL
    delete_sample(sample["id"], db_path=test_db)
    refreshed_gen = get_generation(gen["id"], db_path=test_db)
    assert refreshed_gen is not None
    assert refreshed_gen["reference_sample_id"] is None
