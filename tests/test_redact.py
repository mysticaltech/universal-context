"""Tests for secret redaction."""

from universal_context.redact import has_secrets, redact_secrets


class TestRedaction:
    def test_api_key_redacted(self):
        text = 'api_key = "sk-1234567890abcdef1234567890"'
        result = redact_secrets(text)
        assert "sk-1234567890" not in result
        assert "[REDACTED:" in result

    def test_bearer_token_redacted(self):
        text = "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.xyz"
        result = redact_secrets(text)
        assert "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "[REDACTED:" in result

    def test_aws_key_redacted(self):
        text = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"
        result = redact_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_github_token_redacted(self):
        text = "GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef1234"
        result = redact_secrets(text)
        assert "ghp_" not in result

    def test_anthropic_key_redacted(self):
        text = "ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
        result = redact_secrets(text)
        assert "sk-ant-" not in result

    def test_openai_key_redacted(self):
        text = "OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz"
        result = redact_secrets(text)
        assert "sk-proj-" not in result

    def test_password_redacted(self):
        text = 'password = "my_super_secret_password"'
        result = redact_secrets(text)
        assert "my_super_secret" not in result

    def test_connection_string_redacted(self):
        text = "DATABASE_URL=postgres://user:secret123@db.example.com:5432/mydb"
        result = redact_secrets(text)
        assert "secret123" not in result

    def test_private_key_redacted(self):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----"
        result = redact_secrets(text)
        assert "BEGIN RSA PRIVATE KEY" not in result

    def test_normal_text_unchanged(self):
        text = "This is just a normal conversation about fixing a bug in auth.py"
        result = redact_secrets(text)
        assert result == text

    def test_has_secrets_true(self):
        assert has_secrets("my api_key = sk-1234567890abcdefghijklmnopqrst") is True

    def test_has_secrets_false(self):
        assert has_secrets("just normal code") is False

    def test_mixed_content(self):
        text = (
            "I set up the database with postgres://admin:p@ss@db:5432/app\n"
            "The auth module is working fine now."
        )
        result = redact_secrets(text)
        assert "admin:p@ss" not in result
        assert "auth module is working fine" in result
