"""Typed GitHub identities used by OpenSteward."""

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)


class StrictGitHubModel(BaseModel):
    """Base model for strict GitHub data validation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class GitHubAccountType(StrEnum):
    """GitHub account types that may own an App installation."""

    USER = "user"
    ORGANIZATION = "organization"


class GitHubRepositoryRef(StrictGitHubModel):
    """Repository identity independent of an API response."""

    owner: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=100)

    @field_validator("owner", "name")
    @classmethod
    def validate_repository_segment(
        cls,
        value: str,
    ) -> str:
        """Ensure owner and repository remain separate path segments."""

        if "/" in value or "\\" in value:
            raise ValueError(
                "GitHub repository segments must not contain slashes."
            )

        if value in {".", ".."}:
            raise ValueError(
                "GitHub repository segments must not be '.' or '..'."
            )

        return value

    @computed_field
    @property
    def full_name(self) -> str:
        """Return the repository's owner/name identifier."""

        return f"{self.owner}/{self.name}"


class GitHubInstallationRef(StrictGitHubModel):
    """Identity of one GitHub App installation."""

    installation_id: int = Field(gt=0)
    account_login: str = Field(min_length=1, max_length=100)
    account_type: GitHubAccountType

    @field_validator("account_login")
    @classmethod
    def validate_account_login(
        cls,
        value: str,
    ) -> str:
        """Reject account values that could become URL paths."""

        if "/" in value or "\\" in value:
            raise ValueError(
                "GitHub account login must not contain slashes."
            )

        return value