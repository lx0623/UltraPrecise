create table load_proc(l_part int, w_id int, primary key(l_part, w_id));

create table warehouse (w_id int
  ,  w_name varchar(10)
  ,  w_street_1 varchar(20)
  ,  w_street_2 varchar(20)
  ,  w_city varchar(20)
  ,  w_state char(2)
  ,  w_zip char(9)
  ,  w_tax decimal(4,4)
  ,  w_ytd decimal(12,2)
  , primary key(w_id));

create table district (d_id int
    , d_w_id int
    , d_name varchar(10)
    , d_street_1 varchar(20)
    , d_street_2 varchar(20)
    , d_city varchar(20)
    , d_state char(2)
    , d_zip char(9)
    , d_tax decimal(4,4)
    , d_ytd decimal(12,2)
    , d_next_o_id int
    , primary key(d_w_id, d_id));

create table customer (c_id int
    ,  c_d_id int
    ,  c_w_id int
    ,  c_first varchar(16)
    ,  c_middle char(2)
    ,  c_last varchar(16)
    ,  c_street_1 varchar(20)
    ,  c_street_2 varchar(20)
    ,  c_city varchar(20)
    ,  c_state char(2)
    ,  c_zip char(9)
    ,  c_phone char(16)
    ,  c_since date
    ,  c_credit char(2)
    ,  c_credit_lim decimal(12, 2)
    ,  c_discount decimal(4, 4)
    ,  c_balance decimal(12, 2)
    ,  c_ytd_payment decimal(12, 2)
    ,  c_payment_cnt int
    ,  c_delivery_cnt int
    ,  c_data varchar(500)
    ,  primary key (c_w_id, c_d_id, c_id) );

create table history (h_c_id int
    , h_c_d_id int
    , h_c_w_id int
    , h_d_id int
    , h_w_id int
    , h_date date
    , h_amount decimal(6, 2)
    , h_data varchar(24));

create table neworder (no_o_id int
    ,  no_d_id int
    ,  no_w_id int
    ,  primary key ( no_w_id,  no_d_id,  no_o_id ));

create table orders (o_id int
    , o_d_id int
    , o_w_id int
    , o_c_id int
    , o_entry_d date
    , o_carrier_id int
    , o_ol_cnt int
    , o_all_local int
    , primary key ( o_w_id, o_d_id, o_id ));

create table orderline (ol_o_id int
    , ol_d_id int
    , ol_w_id int
    , ol_number int
    , ol_i_id int
    , ol_supply_w_id int
    , ol_delivery_d date
    , ol_quantity int
    , ol_amount decimal(6, 2)
    , ol_dist_info char(24)
    , primary key (ol_w_id, ol_d_id, ol_o_id, ol_number ));

create table item (i_id int
    ,  i_im_id int
    ,  i_name varchar(24)
    ,  i_price decimal(5,2)
    ,  i_data varchar(50)
  , primary key(i_id));

create table stock (s_i_id int
    ,  s_w_id int
    ,  s_quantity int
    ,  s_dist_01 char(24)
    ,  s_dist_02 char(24)
    ,  s_dist_03 char(24)
    ,  s_dist_04 char(24)
    ,  s_dist_05 char(24)
    ,  s_dist_06 char(24)
    ,  s_dist_07 char(24)
    ,  s_dist_08 char(24)
    ,  s_dist_09 char(24)
    ,  s_dist_10 char(24)
    ,  s_ytd decimal(8)
    ,  s_order_cnt int
    ,  s_remote_cnt int
    ,  s_data varchar(50)
    ,  primary key (s_w_id, s_i_id));

create table load_hist(
        total_w_cnt int,
        min_w_id int,
        max_w_id int,
        thread_num int,
        dsn varchar(64),
        load_begin varchar(30),
        load_end varchar(30),
        seed_val int,
        c_val int,
        mem_ratio int,
        status int);
create table audit1 (start_d date);

