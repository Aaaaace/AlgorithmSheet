import java.util.List;

public class InsertionSort {

    /**
     * 插入排序
     * 从小到大排序，入参数组排列顺序在调用方法后会改变
     * 时间复杂度：O(N^2)
     * 空间复杂度：O(1)
     * 
     * @param <T> 元素类型
     * @param array 待排序的数组
     */
    public static <T extends Comparable<T>> void sort(List<T> array) {
        T temp;
        for (int i = 1; i < array.size(); i++) {
            for (int j = i; j > 0; j--) {
                if (array.get(j).compareTo(array.get(j - 1)) < 0) {
                    temp = array.get(j);
                    array.set(j, array.get(j - 1));
                    array.set(j - 1, temp);
                } else {
                    break;
                }
            }
        }
    }
}