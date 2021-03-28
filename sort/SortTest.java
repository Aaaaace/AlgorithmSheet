import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 排序算法测试，TODO: 能够检验结果，能够计时，并且自动覆盖所有已实现的排序算法，最好能够计算内存使用情况
 */
public class SortTest {
    public static void main(String[] args) {

        int length = 10000;
        List<Integer> intList = new ArrayList<>();
        Random random = new Random();
        for (int i = 0; i <length; i++){
            intList.add(random.nextInt(2 * length));
        }
        
        // List<Integer> intList = new ArrayList<>(List.of(9, 5, 6, 7, 2, 8, 3, 4, 1, 1));
        QuickSort.sort(intList);
        System.out.println(intList);
        List<Integer> intList2 = new ArrayList<>(List.of(1));
        QuickSort.sort(intList2);
        System.out.println(intList2);
    }
}
