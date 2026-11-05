from sqlalchemy import create_engine, text

from services.accounts import (
    DuplicateEmailError,
    authenticate_account,
    hash_password,
    is_valid_email,
    register_account,
    verify_password,
)


def _account_engine():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE app_users (
                user_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """))
    return engine


def test_passwords_are_salted_and_only_hashes_are_stored():
    password = "test-password-94"
    first_hash = hash_password(password)
    second_hash = hash_password(password)
    assert first_hash != second_hash
    assert password not in first_hash
    assert verify_password(password, first_hash)
    assert not verify_password("wrong-password", first_hash)


def test_registration_normalizes_email_and_rejects_duplicates():
    engine = _account_engine()
    account = register_account(engine, "Casey Example", "Casey@Example.com", "test-password-94")
    assert account["email"] == "casey@example.com"
    try:
        register_account(engine, "Another User", "casey@example.com", "different-password-58")
    except DuplicateEmailError:
        pass
    else:
        raise AssertionError("Duplicate email registration should be rejected")
    with engine.connect() as connection:
        stored_hash = connection.execute(
            text("SELECT password_hash FROM app_users WHERE email=:email"),
            {"email": "casey@example.com"},
        ).scalar_one()
    assert "test-password-94" not in stored_hash
    engine.dispose()


def test_authentication_accepts_correct_password_and_rejects_incorrect_password():
    engine = _account_engine()
    registered = register_account(engine, "Jordan Example", "jordan@example.com", "test-password-72")
    authenticated = authenticate_account(engine, "JORDAN@example.com", "test-password-72")
    assert authenticated["user_id"] == registered["user_id"]
    assert authenticate_account(engine, "jordan@example.com", "incorrect-password") is None
    engine.dispose()


def test_registered_accounts_have_distinct_history_scopes():
    engine = _account_engine()
    first = register_account(engine, "User One", "one@example.com", "password-one-84")
    second = register_account(engine, "User Two", "two@example.com", "password-two-73")
    assert first["user_id"] != second["user_id"]
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE prediction_history (user_id TEXT NOT NULL, result TEXT NOT NULL)"))
        connection.execute(
            text("INSERT INTO prediction_history (user_id,result) VALUES (:user_id,:result)"),
            [{"user_id": first["user_id"], "result": "one-result"}, {"user_id": second["user_id"], "result": "two-result"}],
        )
    with engine.connect() as connection:
        first_results = connection.execute(
            text("SELECT result FROM prediction_history WHERE user_id=:user_id"),
            {"user_id": first["user_id"]},
        ).scalars().all()
        second_results = connection.execute(
            text("SELECT result FROM prediction_history WHERE user_id=:user_id"),
            {"user_id": second["user_id"]},
        ).scalars().all()
    assert first_results == ["one-result"]
    assert second_results == ["two-result"]
    engine.dispose()


def test_registration_validates_email_format():
    assert is_valid_email("person@example.org")
    assert not is_valid_email("person@localhost")
    assert not is_valid_email("not-an-email")
