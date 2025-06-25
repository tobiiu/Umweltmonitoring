from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

def check_table_exists(engine, table_name="sensor_data", schema="public"):
    """Check if table already exists"""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = :schema
                AND table_name = :table_name
            );
        """), {"schema": schema, "table_name": table_name})
        return result.scalar()

def init_db():
    print("🚀 Initializing database...")
    engine = create_engine(
        "postgresql://postgres:postgres@localhost:5433/env_monitoring"
    )

    try:
        if check_table_exists(engine):
            print("⏩ Table already exists - skipping creation")
        else:
            with engine.begin() as conn:
                # Enable Timescale
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))

                # Create table with unique constraint
                conn.execute(text("""
                    CREATE TABLE sensor_data (
                        timestamp TIMESTAMPTZ NOT NULL,
                        box_id TEXT NOT NULL,
                        sensor_id TEXT NOT NULL,
                        measurement DOUBLE PRECISION,
                        unit TEXT,
                        sensor_type TEXT,
                        icon TEXT,
                        -- Composite unique constraint
                        UNIQUE (timestamp, box_id, sensor_id)
                    );
                """))

                # Convert to hypertable
                conn.execute(text("""
                    SELECT create_hypertable(
                        'sensor_data', 
                        'timestamp',
                        if_not_exists => TRUE
                    );
                """))

            print("✅ Created 'sensor_data' hypertable with unique constraints")

    except SQLAlchemyError as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    init_db()