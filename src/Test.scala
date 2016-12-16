import java.util.logging.Logger

/**
  * Created by root on 2016/5/7 0007.
  */

class Test() {
  def getNum(num: Int):Int={
       3+num;
  }
}
object  Test{
  def main(args: Array[String]) {
    Array.fill(10)(1.0).foreach(println(_))
    val test=new Test();
    println(test.getNum(4));
  }
}
