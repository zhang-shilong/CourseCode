create table feedback
(
    id       int unsigned auto_increment
        primary key,
    name     text     null,
    nickname text     null,
    email    text     null,
    tel      text     null,
    feedback text     null,
    time     datetime null
);

INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (1, '张世龙', '', '张世龙', '张世龙', '张世龙', '2022-06-16 03:10:48');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (2, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', '留言测试2', '2022-06-16 03:11:19');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (3, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', '留言测试，时间', '2022-06-16 11:12:11');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (4, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', '测试测试', '2022-06-16 11:21:34');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (5, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', '再测试', '2022-06-16 01:01:37');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (6, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', '时间出错', '2022-06-16 01:02:41');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (7, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', 'Asia/
Shanghai', '2022-06-16 01:03:56');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (8, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', 'H', '2022-06-16 13:05:25');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (9, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', '测试测试测试', '2022-06-16 19:32:43');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (10, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', '11111', '2022-06-16 19:50:12');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (11, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', '44444', '2022-06-16 19:52:38');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (12, '张世龙', 'nickname', 'shilong.zhang.cn@gmail.com', '', 'nickname测试', '2022-06-16 20:50:36');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (13, 'limitless', '这里是昵称', 'zhng_shilong@outlook.com', '', '换行测试
换行测试
第3行', '2022-06-16 20:58:36');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (14, 'limitless', '', 'zhang_shilong@outlook.com', '', '空格测试             空格测试', '2022-06-16 23:27:54');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (15, '张世龙', '小张', 'shilong.zhang.cn@gmail.com', '', 'SELECT * FROM web_class.feedback;', '2022-06-16 23:48:15');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (18, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', '留言', '2022-06-23 09:52:45');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (19, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', '留言测试', '2022-06-24 10:41:13');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (20, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', '留言测试', '2022-06-24 10:42:01');
INSERT INTO web_class.feedback (id, name, nickname, email, tel, feedback, time) VALUES (21, '张世龙', '', 'shilong.zhang.cn@gmail.com', '', 'aaaaa', '2022-06-24 10:47:13');