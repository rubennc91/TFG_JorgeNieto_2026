-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity myFunction_QDLDL_factor_linsys_solver_KKT_i_rom is 
    generic(
             DWIDTH     : integer := 6; 
             AWIDTH     : integer := 7; 
             MEM_SIZE    : integer := 79
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of myFunction_QDLDL_factor_linsys_solver_KKT_i_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "000000", 1 => "000001", 2 => "000010", 3 => "000011", 4 => "000010", 
    5 => "000100", 6 => "000101", 7 => "000100", 8 => "000110", 9 => "000111", 
    10 => "000110", 11 => "000011", 12 => "000111", 13 => "001000", 14 => "001001", 
    15 => "001010", 16 => "001011", 17 => "001010", 18 => "001100", 19 => "001101", 
    20 => "001110", 21 => "001100", 22 => "001101", 23 => "001110", 24 => "001011", 
    25 => "001111", 26 => "010000", 27 => "010001", 28 => "010010", 29 => "010001", 
    30 => "010011", 31 => "010100", 32 => "010011", 33 => "010101", 34 => "010110", 
    35 to 36=> "010111", 37 => "011000", 38 => "010110", 39 => "010101", 40 => "011001", 
    41 => "011010", 42 => "011001", 43 => "011000", 44 => "010010", 45 => "010100", 
    46 => "010110", 47 => "011011", 48 => "010100", 49 => "011010", 50 => "011100", 
    51 => "011101", 52 => "011011", 53 => "011100", 54 => "010000", 55 => "001111", 
    56 => "001110", 57 => "011110", 58 => "001000", 59 => "010000", 60 => "000001", 
    61 => "001111", 62 => "001101", 63 => "001001", 64 => "011110", 65 => "011111", 
    66 => "000110", 67 => "000000", 68 => "000101", 69 => "001001", 70 => "100000", 
    71 => "011001", 72 => "011000", 73 => "000000", 74 => "011110", 75 => "011101", 
    76 => "011111", 77 => "100000", 78 => "100001" );


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

p_rom_access: process (clk)  
begin 
    if (clk'event and clk = '1') then
        if (ce0 = '1') then 
            q0 <= mem(CONV_INTEGER(addr0_tmp)); 
        end if;
    end if;
end process;

end rtl;

Library IEEE;
use IEEE.std_logic_1164.all;

entity myFunction_QDLDL_factor_linsys_solver_KKT_i is
    generic (
        DataWidth : INTEGER := 6;
        AddressRange : INTEGER := 79;
        AddressWidth : INTEGER := 7);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of myFunction_QDLDL_factor_linsys_solver_KKT_i is
    component myFunction_QDLDL_factor_linsys_solver_KKT_i_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    myFunction_QDLDL_factor_linsys_solver_KKT_i_rom_U :  component myFunction_QDLDL_factor_linsys_solver_KKT_i_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


