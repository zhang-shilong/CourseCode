create table login
(
    id       int unsigned auto_increment
        primary key,
    username text null,
    password text null
);

INSERT INTO web_class.login (id, username, password) VALUES (1, 'zsl', 'zslzsl');