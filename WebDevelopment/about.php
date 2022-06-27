<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <title>About - 张世龙的个人博客</title>
    <link href="scripts/style.css" rel="stylesheet" type="text/css">
    <script src="base/jquery-1.12.4.min.js"></script>
    <script type="text/javascript" src="scripts/func.js"></script>
</head>

<body>
    <?php
        require("scripts/utils.php");
        if(!is_session_started()){
            session_start();
        }
    ?>
    <?php include("scripts/sidebar.html"); ?>
    <?php include("scripts/header.php"); ?>
    <!-- 右侧内容区域 -->
    <div class="main-wrapper">
        <div class="main-content">
            <a name="About"></a><h2>About</h2>
            <p>本网站是《基于 Linux 的 Web 开发》的课程设计。</p>
            <p>主要实现了 5 个页面，包括主页、个人简历、博客文章、出版物和关于页面。</p>
            <p>包括以下特色：</p>
            <ul>
                <li>基于 JavaScript 生成的动态目录，会根据文章的 h2 标题自动生成，并展示当前浏览的位置，且目录支持点击跳转；</li>
                <li>支持 Feedback 功能，PHP 与 MySQL 交互，将 Feedback 表单结果存入关系型数据库；</li>
                <li>支持登录和登出功能；</li>
                <li>登录后在 Articles 页面可以发表新文章，使用了开源 Markdown 编辑器 Editor.md；</li>
                <li>Articles 页面的文章被存储在 MySQL 数据库中，点击左侧的文章列表可跳转，同时支持动态目录生成；</li>
                <li>宽度自适应页面，当页面宽度不足时，自动隐藏目录。</li>
            </ul>
            <p>部分内容的实现参考了网络公开资料。</p>
            <p>本项目使用的 JavaScript 开源库：</p>
            <ul>
                <li>jQuery: <a href="https://jquery.com/" class="black-ref">https://jquery.com/</a></li>
                <li>Editor.md: <a href="https://pandao.github.io/editor.md/" class="black-ref">https://pandao.github.io/editor.md/</a></li>
            </ul>
            <p>本项目的环境设置：</p>
            <ul>
                <li>MySQL 数据库默认端口：localhost:3306，用户名：root，密码：root，MySQL 服务未开启或未连接成功时会弹窗提示，该设置可通过修改 utils.php 的 check_table() 函数变量更改；</li>
                <li>MySQL 数据库的 schema 名称为 web_class，其下存在 3 张表：login、feedback 和 articles，数据表未创建时会自动创建；</li>
                <li>登录功能仅支持唯一的个人账户，用户名：zsl，密码：zslzsl。</li>
            </ul>
            <p><b>Code availability</b>：<a class="black-ref" href="https://github.com/zhang-shilong/CourseCode/tree/master/WebDevelopment">https://github.com/zhang-shilong/CourseCode/tree/master/WebDevelopment</a>。</p>

            <a name="Changelog"></a><h2>Changelog</h2>
            <h3>2022-06-28</h3>
            <ol>
                <li>提高了在不同屏幕大小的适配性；</li>
                <li>再次提高了网站输入的安全性。</li>
            </ol>
            <h3>2022-06-26</h3>
            <ol>
                <li>实现了写文章的功能，在登录后可以发表新文章了；</li>
                <li>现在 Articles 页面文章存储在 MySQL 数据库，并从数据库获取文章列表；</li>
                <li>提高了网站输入的安全性。</li>
            </ol>
            <h3>2022-06-19</h3>
            <ol>
                <li>实现了登录和登出的功能；</li>
                <li>实现了 Feedback 表单组件，提交后信息被写入 MySQL 数据库；</li>
                <li>已提交的反馈会展示在 Feedback 表单下方了；</li>
                <li>现在使用了 PHP 的 include 函数，代码不会出现大量重复了；</li>
                <li>修复了点击目录标签不跳转的 bug；</li>
                <li>修复了缩小页面后博客顶端内容显示不全的 bug。</li>
            </ol>
            <h3>2022-06-12</h3>
            <ol>
                <li>添加了 Resume 页面，嵌入个人简历 PDF 文件；</li>
                <li>添加了 Publications 页面，完成页面的设计；</li>
                <li>添加了 About 页面；</li>
                <li>编写了自动目录的 JavaScript 代码，现在可以自动生成目录了；</li>
                <li>编写了目录指示代码，当前浏览章节的目录有底色了。</li>
            </ol>
            <h3>2022-06-05</h3>
            <ol>
                <li>添加了 Main 页面，完成了页面的基础设计；</li>
                <li>完成了宽度自适应需求，页面宽度不足时自动隐藏目录。</li>
            </ol>

            <a name="Feedback"></a><h2>Feedback</h2>
            <form method="post" action="scripts/submit_feedback.php">
                <table style="width: 100%">
                    <tr align="left">
                        <td>姓名&nbsp;&nbsp;：<input name="name" type="text" class="feedback-text"></td>
                        <td>昵称：<input name="nickname" type="text" class="feedback-text"></td>
                    </tr>
                    <tr align="left">
                        <td>邮箱<span style="color: red">*</span>：<input name="email" type="email" class="feedback-text" required></td>
                        <td>手机：<input name="tel" type="text" class="feedback-text"></td>
                    </tr>
                    <tr align="left">
                        <td colspan="2">留言<span style="color: red">*</span>：<textarea name="feedback" class="feedback-textarea" required></textarea>
                    </tr>
                </table>
                <center>
                    <p><input type="submit" name="submit" value="提交" id="feedback-button" onclick="submitButton()"/></p>
                </center>
            </form>

            <?php
                $db_name = "web_class";
                $table_name = "feedback";
                $sql_create_table = "
                    CREATE TABLE `$table_name`
                    (
                        `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                        `name` TEXT,
                        `nickname` TEXT,
                        `email` TEXT NOT NULL,
                        `tel` TEXT,
                        `feedback` TEXT NOT NULL,
                        `time` DATETIME
                    );";
                $link = check_table($db_name, $table_name, $sql_create_table);

                $sql_query = "SELECT * FROM `$table_name` ORDER BY `id` DESC;";
                $result_query = $link->query($sql_query);
                $rows = $result_query->num_rows;
                if ($rows != 0) {
                    $page_size = 5;
                    $page_count = ceil($rows / $page_size);
                    if (!isset($_GET["page_no"])){
                        $page_no = 1;
                    }
                    else {
                        $page_no = $_GET["page_no"];
                    }
                    if ($page_no > $page_count) {
                        $page_no = $page_count;
                    }
                    $offset = ($page_no - 1) * $page_size;
                    $result_query->data_seek($offset);

                    $i = 0;
                    while ($row = $result_query->fetch_object()) {
                        ?>
                        <div>
                            <hr class="hr100">
                            <?php echo(strtr($row->feedback, array("\n" => "<br>", " " => "&nbsp;"))); ?><br>
                            <span class="feedback-info"><?php echo($row->email); ?></span>
                            <?php
                                if ($row->nickname) {
                            ?>
                            <span class="feedback-info">&nbsp;(<?php echo($row->nickname); ?>)</span>
                            <?php
                                }
                            ?>
                            <span class="feedback-info" style="float: right"><?php echo($row->time); ?></span>
                        </div>
                    <?php
                        $i = $i + 1;
                        if ($i == $page_size) {
                            break;
                        }
                    }
                    ?>
                    <hr class="hr100">
                    <div align="center">
                        [第<?php echo($page_no); ?>页 / 共<?php echo($page_count); ?>页]&nbsp;
                        <?php
                            $url = $_SERVER["REQUEST_URI"];
                            $url = parse_url($url);
                            $url = $url["path"];
                            for ($i = 1; $i <= $page_count; $i++) {
                                echo "<a href=$url?page_no=$i class='black-ref'>$i</a> &nbsp";
                            }
                        ?>
                        [共<?php echo($rows); ?>条记录]
                    </div>
                    <br>
                    <?php
                }
            ?>

        </div>
        <?php include("scripts/catalog.html"); ?>
    </div>

</body>
</html>
