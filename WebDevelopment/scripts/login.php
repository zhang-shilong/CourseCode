<?php
    require("utils.php");
    if(!is_session_started()){
        session_start();
    }

    if (!isset($_SESSION["username"])) {
        $username = $_POST["username"];
        $password = $_POST["password"];
        if (preg_match("/[\'.,:;*?~`!@#$%^&*+=)(<>{}]|\]\[|\/|\\\|\"|\|/", $username) ||
            preg_match("/[\'.,:;*?~`!@#$%^&*+=)(<>{}]|\]\[|\/|\\\|\"|\|/", $password)) {
            echo("<script>alert('用户名或密码不能包含特殊字符！')</script>");
        }
        else {
            $db_name = "web_class";
            $table_name = "login";
            $sql_create_table = "
                CREATE TABLE `$table_name`
                (
                    `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                    `username` TEXT,
                    `password` TEXT
                );";
            $link = check_table($db_name, $table_name, $sql_create_table);

            $sql_query = "SELECT * FROM `$table_name` WHERE `username`='$username' AND `password`='$password'";
            $res_query = $link->query($sql_query);
            $row = mysqli_fetch_array($res_query);

            if (mysqli_num_rows($res_query) == 1) {
                $_SESSION['username'] = $row['username'];
            }
            else {
                echo("<script>alert('用户名或密码错误！')</script>");
            }
        }
        echo("<script>history.go(-1);</script>");
    }

?>
