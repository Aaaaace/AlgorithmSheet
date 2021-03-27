import java.util.List;

public class QuickSort {

    /**
     * 快速排序
     * 从小到大排序，入参数组排列顺序在调用方法后会改变
     * 时间复杂度：O(Nlog2(N))
     * 空间复杂度：O(log(N))  主要来自递归堆栈
     * 
     * @param <T> 元素类型
     * @param array 待排序的数组
     */
    public static <T extends Comparable<T>> void sort(List<T> array) {
        quickSort(array, 0, array.size());
    }

    /**
     * 对划分好的两个部分再递归进行区块划分
     * 
     * @param <T> 元素类型
     * @param array 待排序数组
     * @param left 左边界（包含）
     * @param right 右边界（不包含）
     */
    private static <T extends Comparable<T>> void quickSort(List<T> array, int left, int right) {
        if (right - left < 1) {
            return;
        } else {
            int mid = partition(array, left, right);
            quickSort(array, left, mid);
            quickSort(array, mid + 1, right);
        }
    }

    /**
     * 将一段数组划分为两部分，左半部分小于某个标准数，右半部分大于该标准数
     * 
     * @param <T> 元素类型
     * @param array 待排序数组
     * @param left 左边界（包括）
     * @param right 右边界（不包括）
     * @return 标准数的位置
     */
    private static <T extends Comparable<T>> int partition(List<T> array, int left, int right) {
        T temp = array.get(left);
        array.set(left, null);
        int low = left;
        int high = right - 1;

        // 挖坑填坑方法
        while (low < high) {
            while (low < high && array.get(high).compareTo(temp) >= 0) {
                high--;
            }
            if (low < high) {
                array.set(low, array.get(high));
                low++;
            }
            while (low < high && array.get(low).compareTo(temp) < 0) {
                low++;
            }
            if (low < high) {
                array.set(high, array.get(low));
                high--;
            }
        }

        // 填坑
        array.set(low, temp);

        return low;
    }
}
