-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity myFunction_is_primal_infeasible_Adata_i_rom is 
    generic(
             DWIDTH     : integer := 5; 
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


architecture rtl of myFunction_is_primal_infeasible_Adata_i_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "00000", 1 => "00011", 2 => "00001", 3 => "00100", 4 => "00010", 
    5 => "00100", 6 => "00101", 7 => "00011", 8 => "00110", 9 => "00100", 
    10 => "00111", 11 => "00101", 12 => "00111", 13 => "01000", 14 => "00110", 
    15 => "00111", 16 => "01000", 17 => "01001", 18 => "01111", 19 => "10000", 
    20 => "01010", 21 => "01111", 22 => "00011", 23 => "01011", 24 => "01100", 
    25 => "01111", 26 => "10000", 27 => "10001", 28 => "10010", 29 => "00100", 
    30 => "00101", 31 => "01011", 32 => "01111", 33 => "10001", 34 => "00110", 
    35 => "01101", 36 => "01110", 37 => "10001", 38 => "10010", 39 => "00111", 
    40 => "01000", 41 => "01101", 42 => "10001" );


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

entity myFunction_is_primal_infeasible_Adata_i is
    generic (
        DataWidth : INTEGER := 5;
        AddressRange : INTEGER := 43;
        AddressWidth : INTEGER := 6);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of myFunction_is_primal_infeasible_Adata_i is
    component myFunction_is_primal_infeasible_Adata_i_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    myFunction_is_primal_infeasible_Adata_i_rom_U :  component myFunction_is_primal_infeasible_Adata_i_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


