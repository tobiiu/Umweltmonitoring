CREATE TABLE IF NOT EXISTS sensor_data (
    timestamp TIMESTAMP NOT NULL,
    box_id TEXT NOT NULL,
    sensor_id TEXT NOT NULL,
    measurement TEXT,
    unit TEXT,
    sensor_type TEXT,
    icon TEXT,
    PRIMARY KEY (timestamp, box_id, sensor_id)
);
