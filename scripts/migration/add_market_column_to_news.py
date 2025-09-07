"""add market column to news"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "2025_add_market_to_news"
down_revision = None  # thay bằng revision trước đó
branch_labels = None
depends_on = None

def upgrade():
    market_enum = sa.Enum("vn", "global", name="marketenum")
    market_enum.create(op.get_bind(), checkfirst=True)
    op.add_column("news", sa.Column("market", market_enum, nullable=False, server_default="global"))

def downgrade():
    op.drop_column("news", "market")
    sa.Enum(name="marketenum").drop(op.get_bind(), checkfirst=True)
