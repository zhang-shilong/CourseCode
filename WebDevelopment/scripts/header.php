<!-- 上方导航栏 -->
<div class="top-nav-wrapper">
    <ul class="top-nav-content">
        <li class="top-nav-li"><b><a class="black-ref" href="index.php">Shilong Zhang's Blog</a></b></li>
        <li class="top-nav-li"><a class="top-nav" href="index.php">Main</a></li>
        <li class="top-nav-li"><a class="top-nav" href="resume.php">Résumé</a></li>
        <li class="top-nav-li"><a class="top-nav" href="articles.php">Articles</a></li>
        <li class="top-nav-li"><a class="top-nav" href="publications.php">Publications</a></li>
        <li class="top-nav-li"><a class="top-nav" href="about.php">About</a></li>
    </ul>
    <?php
        if (isset($_SESSION["username"])) {
            ?>
            <div style="position: absolute; top: 20px; right: 130px;">
                hello, <?php echo($_SESSION["username"]) ?>
            </div>
            <div style="position: absolute; top: 0; right: 30px;">
                <form method="post" action="scripts/logout.php">
                    <input type="submit" value="Logout" class="simple-button" onclick="submitButton()" />
                </form>
            </div>
            <?php
        }
        else {
            ?>
        <div style="position: absolute; top: 0; right: 30px;">
            <input type="button" value="Login" class="simple-button" onclick="displayLoginWindow()" />
        </div>
        <?php
        }
    ?>

</div>

<!-- 上方位置指示栏 -->
<div class="top-location-wrapper">
    <div class="top-location-content">
        Your Location:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <?php
            echo($_SERVER['REQUEST_URI']);
        ?>
    </div>
</div>

<!-- 展示登录框 -->
<div id="login-window-id" class="login-window">
    <div style="position: absolute; top: 0; right: 20px; font-size: xx-large">
        <a href="javascript:void(0)" onclick="hideLoginWindow()" class="black-ref">×</a>
    </div>
    <div style="text-align: center; margin-top: 70px;">
        <p style="font-size: xx-large">LOGIN to intranet</p>
        <form method="post" action="scripts/login.php">
            <table style="width: 90%; margin-left: 25%; margin-right: 20%;">
                <tr align="left">
                    <td style="padding-top: 15px;">用户名：<input name="username" type="text" class="login-text" required></td>
                </tr>
                <tr align="left">
                    <td style="padding-top: 20px;">密&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;码：<input name="password" type="password" class="login-text" required></td>
                </tr>
            </table>
            <center>
                <p><input type="submit" name="submit" value="登录" id="login-button" onclick="submitButton()" style="margin-top: 10px;" /></p>
            </center>
        </form>
    </div>
</div>
<div id="shadow" class="shadow-window"></div>
