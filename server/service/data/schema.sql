CREATE TABLE IF NOT EXISTS user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    password TEXT NOT NULL,
    email TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS imgdata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inputname TEXT NOT NULL,
    inputimg TEXT NOT NULL,
    outputimg TEXT NULL,
    contours TEXT NULL,
    upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    seg_time TEXT NULL,
    user_id INTEGER,
    FOREIGN KEY (user_id) REFERENCES user(id)
);