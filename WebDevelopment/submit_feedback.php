<?php
    require("utils.php");

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

    $name = htmlspecialchars($_POST["name"]);
    $nickname = htmlspecialchars($_POST["nickname"]);
    $email = $_POST["email"];
    $tel = htmlspecialchars($_POST["tel"]);
    $feedback = htmlspecialchars($_POST["feedback"]);
    date_default_timezone_set("Asia/Shanghai");
    $date = date("Y/m/d H:i:s", time());
    $sql_insert = "
        INSERT INTO `$table_name`
        VALUES(NULL, '$name', '$nickname', '$email', '$tel', '$feedback', '$date');
    ";
    if ($link->query($sql_insert)) {
        echo("<script>alert('Thanks for your feedback!');</script>");
        header("refresh:0; url='about.php'");
    }
    else {
        echo("<script>alert('[Error] Fail to insert data.');</script>");
        header("refresh:0; url='about.php'");
    }

?>
