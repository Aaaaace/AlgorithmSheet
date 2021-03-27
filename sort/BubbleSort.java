import java.util.List;

public class BubbleSort {

    /**
     * 冒泡排序
     * 从小到大排序，入参数组排列顺序在执行方法后会改变
     * 时间复杂度：O(N^2)
     * 空间复杂度：O(1)
     * 
     * @param <T> 元素类型
     * @param array 待排序的数组
     */
    public static <T extends Comparable<T>> void sort(List<T> array) {
        T temp;
        for (int i = 0; i < array.size(); i++) {
            for (int j = 0; j < array.size() - i - 1; j++) {
                if (array.get(j).compareTo(array.get(j + 1)) > 0) {
                    temp = array.get(j + 1);
                    array.set(j + 1, array.get(j));
                    array.set(j, temp);
                }
            }
        }
    }
}