import java.nio.charset.Charset;
import java.io.*;
import java.util.Properties;
import java.sql.*;
public class jdbc_prepared_stmt_test {
	// public static String driver = "com.mysql.cj.jdbc.Driver"; // connectorj 8.*
	public static String driver = "com.mysql.jdbc.Driver"; // connectorj 5.*
	public static String url = "jdbc:mysql://127.0.0.1:3306/scale_1";
	public static String usr = "root";
	public static String pwd = "hlmxyj123";

	public static void main (String [] args) {
		Connection con;
		String sql;
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
            props.setProperty("user", "root");
            props.setProperty("password", "hlmxyj123");
            props.setProperty("useServerPrepStmts", "true");
			con = DriverManager.getConnection (url, props);

            ///////////////////////////////////
            System.out.println("===================test 1: integer param");
            sql = "select n_nationkey, n_name from NATION where n_nationkey = ?";
            PreparedStatement statement = con.prepareStatement(sql);

            statement.setInt(1, 2);
            rs = statement.executeQuery();
            System.out.println("result:");
            System.out.println("n_nationkey, n_name");
			while (rs.next()) {
				System.out.print(rs.getInt (1)); System.out.print("\t");
				System.out.println (rs.getString (2));
			}

            System.out.println("-------------------test 1: integer param, re-execute");
            rs = statement.executeQuery();
            System.out.println("result again:");
            System.out.println("n_nationkey\tn_name");
			while (rs.next()) {
				System.out.print (rs.getInt (1));
				System.out.print ("\t");
				System.out.println (rs.getString (2));
			}
            statement.close();

            System.out.println();

            System.out.println("===================test 2: null param");
            sql = "select ?";
            statement = con.prepareStatement(sql);
            statement.setNull(1, Types.NULL);
            rs = statement.executeQuery();

            System.out.println("result:");
			while (rs.next()) {
				System.out.println (rs.getString (1));
			}
            statement.close();
            System.out.println();

            // for mysql test data:
            // insert into LINEITEM values (1,1, 1, 1, 100.1, 200.1, 10.1, 20.1, 0, 1, '20190101', '20190101', '20190102', "na", "na", "comment");
            // insert into LINEITEM values (2,2, 2, 2, 200.1, 300.1, 20.1, 30.1, 1, 0, '20190201', '20190201', '20190202', "na", "na", "sdfag iously along the courts. abcd");
            System.out.println("===================test 3: string param, send_longdata");
            String s = "%iously along the cour%";
            sql = "select l_comment from LINEITEM where l_comment like ?";
            statement = con.prepareStatement(sql);
            InputStream inputStream = new ByteArrayInputStream(s.getBytes(Charset.forName("UTF-8")));
            statement.setBlob(1, inputStream);
            rs = statement.executeQuery();

            System.out.println("result:");
			while (rs.next()) {
				System.out.println (rs.getString (1));
			}

            System.out.println("-------------------test 3: string param, send_longdata, re-execute(longdata was cleared)");
            rs = statement.executeQuery();
            System.out.println("result again:");
			while (rs.next()) {
				System.out.println (rs.getString (1));
			}

            System.out.println("-------------------test 3: string param, send_longdata, re-execute, reset blob data");
            s = "%iously along the courts. f%";
            inputStream = new ByteArrayInputStream(s.getBytes(Charset.forName("UTF-8")));
            statement.setBlob(1, inputStream);
            rs = statement.executeQuery();
            System.out.println("result again:");
            System.out.println("l_comment");
			while (rs.next()) {
				System.out.println (rs.getString (1));
			}
            statement.close();
            System.out.println();

            System.out.println("===================test 4: double param");
            sql = "select PS_PARTKEY, PS_SUPPLYCOST from PARTSUPP where PS_SUPPLYCOST < ? ";
            statement = con.prepareStatement(sql);

            statement.setDouble(1, 500.1);
            rs = statement.executeQuery();

            System.out.println("result:");
            System.out.println("PS_PARTKEY\tPS_SUPPLYCOST");
			while (rs.next()) {
			 	System.out.print(rs.getInt (1));
                System.out.print("\t");
			 	System.out.println (rs.getDouble (2));
			}
            statement.close();

            System.out.println("===================test 5: constant param");
            sql = "select ?, ?, PS_PARTKEY, PS_SUPPLYCOST from PARTSUPP where PS_SUPPLYCOST < ?  and PS_AVAILQTY < ?";
            statement = con.prepareStatement(sql);

            statement.setDouble(1, 100.1);
            statement.setInt(2, 999);
            statement.setDouble(3, 100.1);
            statement.setInt(4, 200);
            rs = statement.executeQuery();

            System.out.println("result:");
            System.out.println("?, ?, PS_PARTKEY, PS_SUPPLYCOST");
			while (rs.next()) {
			 	System.out.print(rs.getDouble (1)); System.out.print("\t");
			 	System.out.print(rs.getInt (2)); System.out.print("\t");
			 	System.out.print(rs.getInt (3)); System.out.print("\t");
			 	System.out.println (rs.getDouble (4));
			}
            statement.close();

            // normal sql statements
            // Statement stmt = con.createStatement ();
            // String likeQuery = "select l_comment from lineitem where l_comment like '%iously along the cour%'";
			// rs = stmt.executeQuery (likeQuery);
			// while (rs.next ()) {
			// 	System.out.println (rs.getString (1));
			// }
            // stmt.close();

			con.close();
		} catch (SQLException e) {
			System.out.println ("error!SQL Exception:" + e);
			return;
		}
	}
}
