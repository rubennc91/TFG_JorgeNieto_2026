-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity myFunction_QDLDL_factor_linsys_solver_KKT_p_rom is 
    generic(
             DWIDTH     : integer := 7; 
             AWIDTH     : integer := 6; 
             MEM_SIZE    : integer := 35
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          addr1      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce1       : in std_logic; 
          q1         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of myFunction_QDLDL_factor_linsys_solver_KKT_p_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
signal addr1_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "0000000", 1 => "0000001", 2 => "0000010", 3 => "0000011", 
    4 => "0000101", 5 => "0000110", 6 => "0000111", 7 => "0001001", 
    8 => "0001011", 9 => "0001110", 10 => "0001111", 11 => "0010000", 
    12 => "0010010", 13 => "0010011", 14 => "0010100", 15 => "0010111", 
    16 => "0011010", 17 => "0011011", 18 => "0011100", 19 => "0011110", 
    20 => "0011111", 21 => "0100001", 22 => "0100010", 23 => "0100011", 
    24 => "0100100", 25 => "0100110", 26 => "0101001", 27 => "0101100", 
    28 => "0110000", 29 => "0110011", 30 => "0111000", 31 => "1000000", 
    32 => "1000110", 33 => "1001010", 34 => "1001111" );


begin 


memory_access_guard_0: process (addr0) 
begin
      addr0_tmp <= addr0;
--synthesis translate_off
      if (CONV_INTEGER(addr0) > mem_size-1) then
           addr0_tmp <= (others => '0');
      else 
           addr0_tmp <= addr0;
      end if;
--synthesis translate_on
end process;

memory_access_guard_1: process (addr1) 
begin
      addr1_tmp <= addr1;
--synthesis translate_off
      if (CONV_INTEGER(addr1) > mem_size-1) then
           addr1_tmp <= (others => '0');
      else 
           addr1_tmp <= addr1;
      end if;
--synthesis translate_on
end process;

p_rom_access: process (clk)  
begin 
    if (clk'event and clk = '1') then
        if (ce0 = '1') then 
            q0 <= mem(CONV_INTEGER(addr0_tmp)); 
        end if;
        if (ce1 = '1') then 
            q1 <= mem(CONV_INTEGER(addr1_tmp)); 
        end if;
    end if;
end process;

end rtl;

Library IEEE;
use IEEE.std_logic_1164.all;

entity myFunction_QDLDL_factor_linsys_solver_KKT_p is
    generic (
        DataWidth : INTEGER := 7;
        AddressRange : INTEGER := 35;
        AddressWidth : INTEGER := 6);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0);
        address1 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce1 : IN STD_LOGIC;
        q1 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of myFunction_QDLDL_factor_linsys_solver_KKT_p is
    component myFunction_QDLDL_factor_linsys_solver_KKT_p_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR;
            addr1 : IN STD_LOGIC_VECTOR;
            ce1 : IN STD_LOGIC;
            q1 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    myFunction_QDLDL_factor_linsys_solver_KKT_p_rom_U :  component myFunction_QDLDL_factor_linsys_solver_KKT_p_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0,
        addr1 => address1,
        ce1 => ce1,
        q1 => q1);

end architecture;


