import java.util.List;

public class SelectionSort {

    /**
     * 选择排序
     * 从小到大排序，入参数组排列顺序在执行方法后会改变
     * 时间复杂度：O(N^2)
     * 空间复杂度：O(1)
     * 
     * @param <T> 元素类型
     * @param array 待排序的数组
     */
    public static <T extends Comparable<T>> void sort(List<T> array) {
        int minIndex;
        T temp;
        for (int i = 0; i < array.size(); i++) {
            minIndex = i;
            for (int j = i + 1; j < array.size(); j++) {
                if (array.get(minIndex).compareTo(array.get(j)) > 0) {
                    minIndex = j;
                }
            }
            temp = array.get(minIndex);
            array.set(minIndex, array.get(i));
            array.set(i, temp);
        }
    }
}