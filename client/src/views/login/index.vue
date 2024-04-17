<script setup lang="ts">
import { reactive } from 'vue'
import { userRegister } from '@/api/user'
import { useUserStore } from '@/stores/userStore';
import { useRouter } from 'vue-router';
import { message } from 'ant-design-vue';
const router = useRouter()

// import axios from 'axios';
const state = reactive({
    isLogin: true,
    emailError: false,
    passwordError: false,
    existed: false,
    passwordTooShort: false,
})
const form = reactive({
    username: '',
    password: '',
    email: ''
})

// login function
const loginWaring = () => {
    message.warning('请完成信息输入');
}
const loginFail = () => {
    message.error('登录失败');
}
const goBack = () => {
    router.go(-1)
}
const login = () => {
    if (form.username != '' && form.password != '') {
        useUserStore().login({ username: form.username, password: form.password })
    } else {
        loginWaring()
    }
}

// register function
const registerSuccess = () => {
    message.success('注册成功');
}
const registerFail = () => {
    message.error('注册失败');
}
const registerWaring = () => {
    message.warning('请完成信息输入');
}
const userAlreadyExisted = () => {
    message.error('用户已存在');
}
const register = () => {
    if (form.username != '' && form.email != '' && form.password != '') {
        if (form.password.length < 6) {
            state.passwordTooShort = true
            return
        } else {
            state.passwordTooShort = false
            const res = userRegister(form.username, form.password, form.email).then(res => {
                // console.log("res of register:", res);
                if (res.code == 200) {
                    registerSuccess()
                    changeType()
                } else if (res.code == 400) {
                    if (res.msg == 'user already existed') {
                        userAlreadyExisted()
                    } else {
                        registerFail()
                    }
                }
            })
        }
    } else {
        registerWaring()
    }
}
state.passwordTooShort = false


// changeType function
const changeType = () => {
    state.isLogin = !state.isLogin
    form.username = ''
    form.password = ''
    form.email = ''
}
</script>

<template>
    <div class="login">
        <div class="contain">
            <div class="big-box" :class="{ active: state.isLogin }">
                <!-- 账号登录 -->
                <div class="big-contain" v-if="state.isLogin">
                    <div class="btitle">账户登录</div>
                    <div class="bform">
                        <input type="text" placeholder="用户名" v-model="form.username">
                        <span class="errTips" v-if="state.emailError">* 邮箱填写错误 *</span>
                        <input type="password" placeholder="密码" v-model="form.password">
                        <span class="errTips" v-if="state.passwordError">* 密码填写错误 *</span>
                    </div>
                    <button class="bbutton" @click="login">登录</button>
                </div>
                <div class="instruction" v-if="state.isLogin">
                    <span>注册登录即表示同意</span>
                    <RouterLink to="/about">&nbsp;用户协议&nbsp;</RouterLink>
                    <span>和</span>
                    <RouterLink to="/about">&nbsp;隐私政策&nbsp;</RouterLink>
                </div>
                <!-- 注册 -->
                <div class="big-contain" key="bigContainRegister" v-else>
                    <div class="btitle">创建账户</div>
                    <div class="bform">
                        <input type="text" placeholder="用户名" v-model="form.username">
                        <span class="errTips" v-if="state.existed">* 用户名已经存在！ *</span>
                        <input type="text" placeholder="邮箱" v-model="form.email">
                        <input type="password" placeholder="密码" v-model="form.password">
                        <span class="errTips" v-if="state.passwordTooShort">* 密码需要大于六位！ *</span>
                    </div>
                    <button class="bbutton" @click="register">注册</button>
                </div>
            </div>

            <div class="small-box" :class="{ active: state.isLogin }">
                <!-- 登录跳转注册 -->
                <div class="small-contain" key="smallContainRegister" v-if="state.isLogin">
                    <div class="stitle">Hello!</div>
                    <p class="scontent">请尽情探索</p>
                    <button class="sbutton" @click="changeType">注册</button>
                </div>
                <!-- 注册跳转登录 -->
                <div class="small-contain" key="smallContainLogin" v-else>
                    <div class="stitle">Welcome!</div>
                    <p class="scontent">欢迎体验</p>
                    <button class="sbutton" @click="changeType">登录</button>
                </div>
            </div>
        </div>
    </div>
</template>

<style lang="scss" scoped>
.login {
    width: 100vw;
    height: 100vh;
    padding: 2rem;
    box-sizing: border-box;
    justify-content: center;

    .contain {
        width: 600px;
        height: 400px;
        position: relative;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: #fff;
        border-radius: 20px;
        // box-shadow: 0 0 3px #f0f0f0,0 0 6px #f0f0f0;
        box-shadow: 10px 10px 10px #d1d9e6, -10px -10px 10px #f9f9f9;

        .big-box {
            width: 70%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 30%;
            transform: translateX(0%);
            transition: all 1s;

            .big-contain {
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;

                .btitle {
                    font-size: 1.5em;
                    font-weight: bold;
                    color: $primary-200;
                }

                .bform {
                    width: 100%;
                    height: 40%;
                    padding: 1.0em 0;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-around;
                    align-items: center;

                    .errTips {
                        display: block;
                        width: 50%;
                        text-align: left;
                        color: red;
                        font-size: 0.7em;
                        margin-left: 1em;
                    }

                    .otherWay {
                        width: 50%;
                        text-align: left;
                        color: #999;
                        font-size: 0.7em;
                        margin-left: 1em;
                    }
                }

                .bform input {
                    width: 50%;
                    height: 30px;
                    border: none;
                    outline: none;
                    border-radius: 10px;
                    padding-left: 2em;
                    background-color: #f0f0f0;
                    box-shadow: inset 2px 2px 4px #d1d9e6, inset -2px -2px 4px #f9f9f9;
                }

                .bbutton {
                    width: 20%;
                    height: 40px;
                    border-radius: 24px;
                    border: none;
                    outline: none;
                    background-color: $primary-200;
                    color: #fff;
                    font-size: 0.9em;
                    cursor: pointer;
                    box-shadow: 8px 8px 16px #d1d9e6, -8px -8px 16px #f9f9f9;
                }

            }

            .instruction {
                font-size: 0.7em;
                color: #999;
                text-align: center;
                position: relative;
                left: 50%;
                transform: translate(-50%, -180%);

                a {
                    color: $primary-200;
                }
            }
        }

        .big-box.active {
            left: 0;
            transition: all 0.5s;
        }

        .small-box {
            width: 30%;
            height: 100%;
            background: linear-gradient(135deg, $primary-300, $primary-100);
            position: absolute;
            top: 0;
            left: 0;
            transform: translateX(0%);
            transition: all 1s;
            border-top-left-radius: inherit;
            border-bottom-left-radius: inherit;

            .small-contain {
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;

                .stitle {
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #fff;
                }

                .scontent {
                    font-size: 0.9em;
                    color: #fff;
                    text-align: center;
                    padding: 2em 3.5em;
                    line-height: 1.7em;
                }

                .sbutton {
                    width: 60%;
                    height: 40px;
                    border-radius: 24px;
                    border: 1px solid #fff;
                    outline: none;
                    background-color: transparent;
                    color: #fff;
                    font-size: 0.9em;
                    cursor: pointer;

                }
            }
        }

        .small-box.active {
            left: 100%;
            border-top-left-radius: 0;
            border-bottom-left-radius: 0;
            border-top-right-radius: inherit;
            border-bottom-right-radius: inherit;
            transform: translateX(-100%);
            transition: all 1s;
        }
    }
}
</style>