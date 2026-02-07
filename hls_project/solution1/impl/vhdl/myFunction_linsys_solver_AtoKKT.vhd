-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity myFunction_linsys_solver_AtoKKT_rom is 
    generic(
             DWIDTH     : integer := 7; 
             AWIDTH     : integer := 6; 
             MEM_SIZE    : integer := 43
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of myFunction_linsys_solver_AtoKKT_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "0000100", 1 => "0001011", 2 => "0011101", 3 => "0101100", 
    4 => "0100000", 5 => "0101101", 6 => "0110000", 7 => "0001100", 
    8 => "0001010", 9 => "0101110", 10 => "0100110", 11 => "0110001", 
    12 => "0101010", 13 => "0101011", 14 => "0000111", 15 => "0100111", 
    16 => "0100100", 17 => "0010101", 18 => "0010111", 19 => "0010110", 
    20 => "0010001", 21 => "0011000", 22 => "0111010", 23 => "0111011", 
    24 => "0111100", 25 => "0111101", 26 => "0111110", 27 => "1001010", 
    28 => "0111111", 29 => "0110100", 30 => "0110101", 31 => "0110110", 
    32 => "0110111", 33 => "1001011", 34 => "1000010", 35 => "1000011", 
    36 => "1000100", 37 => "1001100", 38 => "1000101", 39 => "1000111", 
    40 => "1001000", 41 => "1001001", 42 => "1001101" );


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

entity myFunction_linsys_solver_AtoKKT is
    generic (
        DataWidth : INTEGER := 7;
        AddressRange : INTEGER := 43;
        AddressWidth : INTEGER := 6);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of myFunction_linsys_solver_AtoKKT is
    component myFunction_linsys_solver_AtoKKT_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    myFunction_linsys_solver_AtoKKT_rom_U :  component myFunction_linsys_solver_AtoKKT_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


