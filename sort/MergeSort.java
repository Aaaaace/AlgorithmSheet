import java.util.ArrayList;
import java.util.List;

public class MergeSort {

    // 对某段数组进行归并排序的最小数组长度，不足时使用插入排序
    private final static int MIN_MERGE_LENGTH = 7;

    /**
     * 归并排序
     * 从小到大排序，入参数组排列顺序在调用方法后会改变
     * 时间复杂度：O(Nlog2(N))
     * 空间复杂度：O(N) 主要来自辅助数组
     * 
     * @param <T> 元素类型
     * @param array 待排序的数组
     */
    public static <T extends Comparable<T>> void sort(List<T> array) {
        List<T> tempArray = new ArrayList<>();
        for(int i = 0; i<array.size(); i++){
            tempArray.add(null);
        }
        mergeSort(array, 0, array.size(), tempArray);
    }

    /**
     * 将数组的一部分拷贝到另一个数组的对应位置
     * 
     * @param <T> 元素类型
     * @param fromArray 源数组
     * @param toArray 目标数组
     * @param left 左边界（包含）
     * @param right 右边界（不包含）
     */
    private static <T extends Comparable<T>> void arraycopy(List<T> fromArray, List<T> toArray, int left, int right) {
        for (int i = left; i < right; i++) {
            toArray.set(i, fromArray.get(i));
        }
    }

    /**
     * 对数组的一个区间进行归并排序
     * 
     * @param <T> 元素类型
     * @param array 待排序数组
     * @param left 左边界（包含）
     * @param right 右边界（不包含）
     * @param tempArray 归并排序时使用的辅助数组，作为全局变量传入，以减少创建开销
     */
    private static <T extends Comparable<T>> void mergeSort(List<T> array, int left, int right, List<T> tempArray) {
        if (right - left < MergeSort.MIN_MERGE_LENGTH) {
            // 插入排序
            T temp;
            for (int i = left; i < right; i++) {
                for (int j = i; j > left; j--) {
                    if (array.get(j).compareTo(array.get(j - 1)) < 0) {
                        temp = array.get(j);
                        array.set(j, array.get(j - 1));
                        array.set(j - 1, temp);
                    } else {
                        break;
                    }
                }
            }
        } else {
            int mid = (left + right) /2 ;
            mergeSort(array, left, mid, tempArray);
            mergeSort(array, mid, right, tempArray);
            mergeOfTwoSortedArray(array, left, mid, right, tempArray);
        }
    }

    /**
     * 归并数组中排序好的两部分
     * 
     * @param <T> 元素类型
     * @param array 待排序数组
     * @param left 左边界（包含）
     * @param mid 中边界（左包含，右不包含）
     * @param right 右边界（不包含）
     * @param tempArray 归并排序时使用的辅助数组，作为全局变量传入，以减少创建开销
     */
    private static <T extends Comparable<T>> void mergeOfTwoSortedArray(List<T> array, int left, int mid, int right,
            List<T> tempArray) {

        int i = left;
        int j = mid;
        int k = left; // 临时数组的索引
        while (i < mid && j < right) {
            if (array.get(i).compareTo(array.get(j)) <= 0) {
                tempArray.set(k, array.get(i));
                i++;
            } else {
                tempArray.set(k, array.get(j));
                j++;
            }
            k++;
        }
        if (i == mid) {
            for (int x = j; x < right; x++) {
                tempArray.set(k++, array.get(x));
            }

        } else {
            for (int x = i; x < mid; x++) {
                tempArray.set(k++, array.get(x));
            }
        }
        arraycopy(tempArray, array, left, right);
    }
}
