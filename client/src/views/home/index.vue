<script setup lang="ts">
import { message } from 'ant-design-vue'
import { uploadImage, getImage } from '@/api/images'
import { ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useUserStore } from '@/stores/userStore';
import { useTimingStore } from '@/stores/timing';

const route = useRoute()
const router = useRouter()
const sourceImg = ref('')
const resultImg = ref('')
const imageID = ref('')
const isScanning = ref(false)
const isEnlarge = ref(false)
const isDetecting = ref(false)
let userId = localStorage.getItem('userInfo') ? JSON.parse(localStorage.getItem('userInfo')!).user_id : ''
const timing = useTimingStore()
const isTiming = ref(false)

// 进行检测按钮事件
const detect = () => {
    if (imageID.value === '') {
        message.error('请先上传图片')
        return
    } else if (isDetecting.value) {
        message.error('正在检测中，请稍后再试')
        return
    } else {
        isDetecting.value = true
        toggleScanning(true)
        getImg()
    }
}
// 向后端请求获取检测结果
const getImg = async () => {
    timing.resetTimer()
    isTiming.value = true
    timing.startTimer()
    const res = await getImage(imageID.value)
    if (res != null) {
        resultImg.value = window.URL.createObjectURL(res)
        timing.pauseTimer()
        toggleScanning(false)
        enlarge()
        isDetecting.value = false
    } else {
        console.log('no res');
        toggleScanning(false)
        isDetecting.value = false
    }
}

// 退出登录
const userStore = useUserStore()
const logout = () => {
    userStore.logout()
    reload()
}
// history record
const record = () => {
    router.push('/record');
}
// reload function(refresh page)
const reload = () => {
    window.location.reload()
}
// 结果图片放大
const enlarge = () => {
    toggleEnlarge(true)
    setTimeout(() => {
        toggleEnlarge(false)
    }, 3000)
}
const toggleEnlarge = (flag: boolean) => {
    isEnlarge.value = flag
}

// 上传图片
const handleUpload = async ({
    target,
}: {
    target: HTMLInputElement & EventTarget
}) => {
    if (isDetecting.value) {
        message.error('正在检测中，请稍后再试')
        return
    } else {
        const reader = new FileReader()
        const files: FileList | null = target.files
        if (files) {
            reader.readAsDataURL(files[0])
            reader.onload = (e) => {
                sourceImg.value = e.target!.result as string
                resultImg.value = ''
            }

            const formData = new FormData()
            formData.append('file', files[0])
            formData.append('user_id', userId)
            isTiming.value = false
            try {
                const res = await uploadImage(formData)
                imageID.value = res.data.image_id
            } catch {
                message.error('上传失败')
            }
            target.value = ''
        }
    }
}


const toggleScanning = (flag: boolean) => {
    isScanning.value = flag
}

watch(
    () => route.path,
    (val) => {
        if (val === '/unload-area-img' || val === '/refuel-area-img') {
            toggleScanning(false)
            sourceImg.value = ''
            resultImg.value = ''
        }
    }
)
</script>

<template>
    <div class="home">
        <div class="header">
            <div class="left">
                <div class="logo"></div>
                <div class="title">眼脉络膜分割智慧诊查系统</div>
            </div>
            <div class="right">
                <div class="avatar">
                    <div class="logout" @click="logout">
                        退出登录
                    </div>
                    <div class="record" @click="record">
                        历史记录
                    </div>
                </div>

            </div>
        </div>
        <div class="photo">
            <p class="img-tip">
                说明：图片处理功能，需要上传一张图片，然后点击检测按钮，等待结果检出；若重复检测相同图片，则直接显示上次检测结果。
            </p>

            <div v-if="isTiming" class="timing">
                检测时间：{{ timing.second }} : {{ timing.millisecond }}
            </div>

            <div class="image-box" style="display: flex">
                <div class="img-input" :style="{ backgroundImage: `url(${sourceImg})` }">
                    <div :class="{ scan: isScanning }"></div>
                </div>
                <div class="img-output" :class="{ enlarge: isEnlarge }"
                    :style="{ backgroundImage: `url(${resultImg})` }">
                </div>
                <div class="background"></div>
                <div class="enlarge"></div>
                <div class="scan"></div>
            </div>

            <div class="btn">
                <button class="button upload-button">
                    上传检测图片
                    <input type="file" accept="image/*" class="file-input" @change="handleUpload" />
                </button>
                <button class="button detect" @click="detect">
                    进行检测
                </button>
            </div>
        </div>


    </div>
</template>

<style lang="scss" scoped>
.home {
    display: flex;
    align-items: center;
    flex-direction: column;
    // background: linear-gradient(to right top, #DDDDDD, #FFFFFF);
    background: linear-gradient(to right top, #dbf1fa, #f8fcff);
    background-blend-mode: multiply;
    color: var(--text-color1);
    min-height: calc(100vh);
    overflow: hidden;

    .header {
        height: 100px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: space-between;

        .left {
            height: 100%;
            display: flex;
            align-items: center;
            padding-left: 200px;
            gap: 25px;

            .logo {
                height: 60px;
                width: 6ch;
                left: 0;
                background: url('@/assets/images/logo.png') no-repeat center 0px / contain;
            }

            .title {
                font-size: 20px;
                font-weight: bold;
            }
        }

        .right {
            height: 100%;
            display: flex;
            align-items: center;
            padding-right: 200px;

            .avatar {
                height: 60px;
                width: 60px;
                background: url('@/assets/images/avatar.png') no-repeat center 0px / contain;
                position: relative;

                .logout {
                    font-size: 16px;
                    cursor: pointer;
                    display: none;
                    width: 90px;
                    height: 34px;
                    background-color: var(---bg1);
                    z-index: 1;
                    border-radius: 5px;
                    position: absolute;
                    top: 149%;
                    left: 50%;
                    transform: translateX(-50%);
                    text-align: center;
                    line-height: 34px;
                    border: 1px solid var(--warning1);
                }

                .record {
                    font-size: 16px;
                    cursor: pointer;
                    display: none;
                    width: 90px;
                    height: 34px;
                    background-color: var(---bg1);
                    z-index: 1;
                    border-radius: 5px;
                    position: absolute;
                    top: 90%;
                    left: 50%;
                    transform: translateX(-50%);
                    text-align: center;
                    line-height: 34px;
                    border: 1px solid var(--border2);
                }
            }

            .avatar:hover .logout {
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .avatar:hover .record {
                display: flex;
                justify-content: center;
                align-items: center;
            }
        }
    }

    .photo {
        width: 1040px;

        .img-tip {
            font-size: 16px;
            color: var(--text-color1);
            margin: 0px 0px 10px 20px;
        }

        .timing {
            font-size: 16px;
            color: var(--text-color1);
            position: fixed;
            left: 45.5%;
        }

        .image-box {
            overflow: hidden;
            position: relative;
            margin-top: 38px;

            .img {
                &-input {
                    flex: 1 0 475px;
                    min-height: 64vh;
                    border-radius: 20px;
                    background-color: var(--bg1);
                    background-repeat: no-repeat;
                    background-size: contain;
                    background-position: center;
                    box-shadow: 2px 5px 5px rgb(0 0 0 / 20%);
                    margin: 5px 20px 10px 5px;
                    position: relative;
                    overflow: hidden;
                }

                &-output {
                    flex: 1 0 475px;
                    min-height: 64vh;
                    border-radius: 20px;
                    background-color: var(--bg1);
                    background-repeat: no-repeat;
                    background-size: contain;
                    background-position: center;
                    // box-shadow: 2px 5px 5px rgb(0 0 0 / 20%);
                    margin: 5px 5px 10px 20px;
                    position: relative;
                    z-index: 1;
                }
            }

            .background {
                position: absolute;
                top: 0;
                border-radius: 20px;
                left: calc(50% + 20px);
                top: calc(0% + 5px);
                width: calc(50% - 25px);
                height: calc(100% - 15px);
                background: var(--bg1);
                box-shadow: 2px 5px 5px rgb(0 0 0 / 20%);
                z-index: 0;
            }

            .enlarge {
                animation: enlarge 3s infinite ease-in-out;
            }

            @keyframes enlarge {
                0% {
                    transform: scale(0.05);
                }

                100% {
                    transform: scale(1);
                }
            }

            .scan {
                position: relative;
                background: linear-gradient(180deg, rgba(0, 255, 51, 0) 43%, #32F0D3 211%);
                height: 100%;
                animation: radar-beam 2s infinite ease-in-out;
            }

            @keyframes radar-beam {
                0% {
                    top: -100%;
                }

                100% {
                    top: 50%;
                }
            }
        }

        .btn {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 250px;
            margin-top: 25px;

            .button {
                font-size: 14px;
                border: 0px;
                height: 40px;
                border-radius: 4px;
                color: var(--text-color2);
                cursor: pointer;
                letter-spacing: 5px;
                font-size: 16px;
                box-shadow: 2px 3px 5px rgb(0 0 0 / 20%);
            }


            .upload-button {
                margin-right: 40px;
                position: relative;
                display: inline-block;
                padding: 10px 20px;
                background-color: #177ddc;
                cursor: pointer;
            }

            input {
                cursor: pointer;
            }

            .file-input {
                position: absolute;
                width: 100%;
                height: 100%;
                top: 0;
                left: 0;
                opacity: 0;
                cursor: pointer;
            }


            .detect {
                background-color: #a61d24;
                background: linear-gradient(to right, var(--primary-100), var(--primary-200));
                padding: 10px 20px;
            }

        }


    }
}
</style>