import java.util.Properties;
import java.sql.*;
public class jdbc_test {
	// public static String driver = "com.mysql.cj.jdbc.Driver"; // connectorj 8.*
	public static String driver = "com.mysql.jdbc.Driver"; // connectorj 5.*
	public static String url = "jdbc:mysql://127.0.0.1:3306/scale_1";
	public static String usr = "root";
	public static String pwd = "123";

	public static void main (String [] args) {
		Connection con;
		Statement sql;
		ResultSet rs;

		try {
            Class.forName("com.mysql.jdbc.Driver").newInstance();
		} catch (ClassNotFoundException e) {
			System.out.println ("error!Class not found");
			return;
		} catch (Exception e) {
			System.out.println ("Excepton");
			return;
		}
		try {
            Properties props = new Properties();
            props.setProperty("user", usr);
            props.setProperty("password", pwd);
            props.setProperty("allowMultiQueries", "true");
			con = DriverManager.getConnection (url, props);
			sql = con.createStatement ();
			rs = sql.executeQuery ("select * from t1; SELECT * FROM nation");
			while (rs.next ()) {
				System.out.println (rs.getString (1));
				System.out.println (rs.getString (2));
			}
			con.close();
		} catch (SQLException e) {
			System.out.println ("error!SQL Exception");
			return;
		}

	}
}
