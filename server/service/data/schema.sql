CREATE TABLE IF NOT EXISTS user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username STRING NOT NULL,
    password STRING NOT NULL,
    email STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS imgdata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inputname TEXT NOT NULL,
    inputimg TEXT NOT NULL,
    outputimg TEXT NULL,
    contours TEXT NULL,
    user_id INTEGER,
    FOREIGN KEY (user_id) REFERENCES user(id)
);