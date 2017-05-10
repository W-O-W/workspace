import java.io.FileInputStream;
import java.io.BufferedInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
public class ReadFileTest {
	public static void main(String[]aegs){
		try{
			FileInputStream file = new FileInputStream("/home/ki/Desktop/test.txt");
			int data = 0;
			while((data = file.read())!=-1){
				System.out.println(data);
			}
			file.close();
		}catch(FileNotFoundException e){
			e.printStackTrace();
		}catch(IOException e){
			e.printStackTrace();
		}
		
		try{
			BufferedInputStream file = new BufferedInputStream(new FileInputStream("/home/ki/Desktop/test.txt"));
			
		}
	}
}
			