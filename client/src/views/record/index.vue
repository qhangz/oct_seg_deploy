<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useUserStore } from '@/stores/userStore';
import { getRecord } from '@/api/images'
import { useRouter } from 'vue-router'

const router = useRouter()
let userId = localStorage.getItem('userInfo') ? JSON.parse(localStorage.getItem('userInfo')!).user_id : ''

interface RecordData {
    output_img: string,
    upload_time: string,
    seg_time: string
}
const recordData = ref<RecordData[]>([])

const getRecordImg = async () => {
    const res = await getRecord(userId)
    recordData.value = res.data.record_data
    for (let i = 0; i < recordData.value.length; i++) {
        recordData.value[i].output_img = "data:image/png;base64," + recordData.value[i].output_img
        recordData.value[i].seg_time = parseFloat(recordData.value[i].seg_time).toFixed(2);
    }
}

const showModal = ref(false)
const modalImage = ref('')
const open = (src: string) => {
    modalImage.value = src
    showModal.value = true
}
const close = () => {
    showModal.value = false
}

const back = () => {
    router.push('/seg')
}

onMounted(() => {
    getRecordImg()
})

</script>

<template>
    <div class="record">
        <div class="header">
            <div class="back" @click="back"></div>
            <div class="title">历史记录</div>
        </div>
        <div class="record-list">
            <div class="container">
                <div class="record-item" v-for="(item, index) in recordData" :key="index">
                    <div class="inner" @click="open(item.output_img)">
                        <img :src="item.output_img" alt="">
                        <div class="message">
                            <div class="upload-time">
                                检测时间：{{ item.upload_time }}
                            </div>
                            <div class="seg-time">
                                检测耗时：{{ item.seg_time }}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div v-if="showModal" class="modal">
            <div class="modal-content">
                <span class="close" @click="close">&times;</span>
                <img :src="modalImage" alt="">
            </div>
        </div>
    </div>
</template>

<style scoped lang="scss">
.record {
    display: flex;
    align-items: center;
    flex-direction: column;
    background: linear-gradient(to right top, #dbf1fa, #f8fcff);
    background-blend-mode: multiply;
    color: var(--text-color1);
    min-height: calc(100vh);
    overflow: hidden;

    .header {
        .back {
            position: absolute;
            left: 15%;
            top: 1.5%;
            width: 55px;
            height: 55px;
            background: url('@/assets/images/back.png') no-repeat center 0px / contain;
            cursor: pointer;
        }

        .title {
            color: var(--text-color1);
            width: 100%;
            height: 50px;
            text-align: center;
            margin-top: 25px;
            font-size: 20px;
            font-weight: bold;
        }
    }

    .record-list {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        position: relative;
        margin: 0 auto;
        width: 100%;
        max-width: 1200px;

        .container {
            width: 100%;
            height: 100%;
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            align-items: stretch;
            margin-left: -7px;
            margin-right: -7px;

            .record-item {

                height: 356px;
                width: 25%;
                padding-left: 7px;
                padding-right: 7px;
                box-sizing: border-box;
                position: relative;
                margin-bottom: 30px;
                color: var(--text-color1);

                .inner {
                    background-color: var(--bg1);
                    box-shadow: 1px 1px 1px rgba(0, 0, 0, .15);
                    transition: all .2s linear;
                    cursor: pointer;
                    padding-bottom: 15px;
                    border-radius: 7px;

                    img {
                        width: 100%;
                        height: 286px;
                    }

                    .message {
                        padding: 15px 5px 0px;
                        background-color: var(--bg1);
                        color: var(--text-color1);
                        font-size: 16px;
                        display: -webkit-box;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        -webkit-box-orient: vertical;
                        -webkit-line-clamp: 2;
                        height: 40px;
                        font-size: 14px;
                        font-weight: 700;
                        text-align: center;
                    }
                }
            }
        }
    }

    .modal {
        position: fixed;
        z-index: 9999;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: rgba(0, 0, 0, 0.5);

        .modal-content {
            max-width: 80%;
            background-color: #fefefe;
            padding: 10px;
            border-radius: 10px;
            position: relative;
        }

        .close {
            position: absolute;
            top: 10px;
            right: 25px;
            color: #aaa;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;

            &:hover,
            &:focus {
                color: rgb(100, 99, 99);
                text-decoration: none;
            }
        }
    }
}
</style>
