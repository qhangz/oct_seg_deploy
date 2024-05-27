import { ref } from 'vue';

export const useTimingStore = () => {
    const second = ref(0);
    const millisecond = ref(0);
    let timerInterval = false;
    let millisecondInterval = false;
    const isPaused = ref(true);

    const startTimer = () => {
        if (!timerInterval) {
            timerInterval = setInterval(() => {
                second.value++;
            }, 1000);
        }
        if (!millisecondInterval) {
            millisecondInterval = setInterval(() => {
                millisecond.value++;
                if (millisecond.value === 1000) {
                    millisecond.value = 0;
                }
            }, 1);
        }
        isPaused.value = false;
    };

    const pauseTimer = () => {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = false;
        }
        if (millisecondInterval) {
            clearInterval(millisecondInterval);
            millisecondInterval = false;
        }
        isPaused.value = true;
    };

    const resetTimer = () => {
        second.value = 0;
        millisecond.value = 0;
    };

    return { second, millisecond, startTimer, pauseTimer, resetTimer, isPaused };
};
