from typing import TYPE_CHECKING

from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.mixins import OrganizationMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.providers import Provider as PydanticProvider

if TYPE_CHECKING:
    from mirix.orm.organization import Organization


class Provider(SqlalchemyBase, OrganizationMixin):
    """Provider ORM class"""

    __tablename__ = "providers"
    __pydantic_model__ = PydanticProvider

    name: Mapped[str] = mapped_column(nullable=False, doc="The name of the provider")
    api_key: Mapped[str] = mapped_column(nullable=True, doc="API key used for requests to the provider.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="providers")
